from itertools import accumulate
import torch
import torch.optim as O
import torch.nn.functional as F

from info_collect import *
import stats
import configs
from circle_ggnn import *
from circle_ggnn_4gru import *
from get_device import *
from get_dataset import *

def total_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {count:,} trainable parameters")

def model_summary(model):
    for name, paras in model.named_parameters():
        print(name, " : ", paras.size())

def softmax_accuracy(probs, all_labels):
    acc = (torch.argmax(probs) == all_labels).sum()
    acc = torch.div(acc, len(all_labels) + 0.0)
    return acc

def right_preds_count(model_preds, true_labels):
    # model_preds = model_preds.to(cpu())
    # right_preds = model_preds.detach().apply_(lambda x: 1.0 if x >= 0.5 else 0.0)
    # right_preds = right_preds.to(device)
    right_preds = (model_preds >= 0.5).int()
    right_preds = (right_preds == true_labels).sum()
    return right_preds

def right_preds_count_multi(model_preds, true_labels):
    model_preds = torch.argmax(model_preds, dim=1)
    # _, y = torch.where(true_labels == 1)
    
    return (model_preds == true_labels).sum()

def test_(model, train_set_loader, device):
    for i, batch in enumerate(train_set_loader):
        if i >= 1:
            return
        batch.to(device)
        b_hat = model(batch)
        b = batch.y
        print(b_hat)
        print(b)
        # print(torch.argmax(b_hat))
        # print(b_hat)
        print(right_preds_count_multi(b_hat, b))

# train our model on train dataset
def train(model, 
          train_set_loader, 
          val_set_loader,
          epochs,
          optimizer,
          loss,
          device):
    history = History("5 classes")
    accumulate_steps = 4
    for epoch in range(epochs):
        train_stats = stats.Stats("train set at epoch " + str(epoch + 1))
        model.train()
        for i, batch in enumerate(train_set_loader):
            batch.to(device)
            x, y = batch, batch.y
            y_hat = model(x)
            l = loss(y_hat, y)
            rps = right_preds_count_multi(y_hat, y)
            l.backward()
            if (i+1) % accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            stat = stats.Stat(y_hat.tolist(), y.tolist(), l.item(), rps.item(), y.numel())
            train_stats(stat)
        history(train_stats, epoch + 1)

        if val_set_loader is not None:
            with torch.no_grad():
                model.eval()
                val_stats = stats.Stats("validate set at epoch " + str(epoch + 1))
                for val_i, val_batch in enumerate(val_set_loader):
                    val_batch.to(device)
                    val_y_hat = model(val_batch)
                    val_l = loss(val_y_hat, val_batch.y)
                    val_rps = right_preds_count_multi(val_y_hat, val_batch.y)
                    val_stat = stats.Stat(val_y_hat.tolist(), val_batch.y.tolist(), val_l.item(), val_rps.item(), val_batch.y.numel())
                    val_stats(val_stat)
                history(val_stats, epoch + 1)
            # history.log()
        
        print(history)

# evaluate our model on test dataset
def eval(model, 
         test_set_loader,
         loss,
         device):
    print(f"evaluate model on test dataset...")
    with torch.no_grad():
        model.eval()
        test_stats = stats.Stats("test process")
        rps, total = 0, 0
        for i, batch in enumerate(test_set_loader):
            batch.to(device)
            y_hat = model(batch)
            # _, y = torch.where(batch.y == 1)
            l = loss(y_hat, batch.y)
            rps += right_preds_count_multi(y_hat, batch.y).item()
            total += batch.y.numel()
            # stat = stats.Stat(y_hat.tolist(), batch.y.tolist(), l.item(), rps.item(), batch.y.numel())
            # test_stats(stat)
        # metrics = Metrics(test_stats.outs(), test_stats.labels())
        # metrics.log()
    # return metrics()
    return rps / total

if __name__ == "__main__":
    all_config = configs.Tensor_GGNN_GCN()
    
    model_config = all_config.model
    circle_ggnn_layer_config = model_config["gated_graph_conv_args"]
    conv_output_layer_config = model_config["conv_args"]
    data_embedding_size = model_config["emb_size"]

    learning_rate = all_config.learning_rate
    weight_decay = all_config.weight_decay
    regular_lambda = all_config.loss_lambda
    batch_size = 128
    epochs = 50
    n_classes = 5

    # print(circle_ggnn_layer_config)
    # print(conv_output_layer_config)
    # print(data_embedding_size)

    run_device = try_device("idle")

    # net = CircleGGNN(circle_ggnn_layer_config, conv_output_layer_config, data_embedding_size)
    # net = CircleGGNN4Gru(circle_ggnn_layer_config, conv_output_layer_config, data_embedding_size)
    net = MultiCircleGGNN(circle_ggnn_layer_config, conv_output_layer_config, data_embedding_size, n_classes=n_classes)
    net.to(device=run_device)

    # total_parameters(net)
    # model_summary(net)
    
    loss_func = lambda o, t: F.cross_entropy(o, t) # + F.los(o, t) * regular_lambda
    target_optim = O.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_set_loader, val_set_loader, test_set_loader = get_multiclass_dataset_loader(batch_size=batch_size, n_classes=n_classes, verbose=True)

    # test_(net, train_set_loader, run_device)

    train(net, 
          train_set_loader,
          val_set_loader, 
          epochs,
          target_optim,
          loss_func,
          run_device)
    
    model_save_path = "./saved_models/circle_ggnn_model_multi"
    net.save(model_save_path)

    eval_metrics = eval(net, test_set_loader, loss_func, run_device)
    print(eval_metrics)
    # metrics = ["Accuracy", "Precision", "Recall", "F-measure"]
    # for m in metrics:
    #     print(m, eval_metrics[m])