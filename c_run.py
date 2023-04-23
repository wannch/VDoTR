import torch
import torch.optim as O
import torch.nn.functional as F

import imports

from models import *

from info_collect import *
from stats import *
from configs import *
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
        b = batch.y.float()
        # print(torch.argmax(b_hat))
        # print(b_hat)
        print(right_preds_count(b_hat, b))

# train our model on train dataset
def train(model, 
          train_set_loader, 
          val_set_loader,
          epochs,
          optimizer,
          loss,
          device):
    history = History("TensorGCN binary class Com")
    for epoch in range(epochs):
        train_stats = Stats("train set at epoch " + str(epoch + 1))
        model.train()
        for i, batch in enumerate(train_set_loader):
            batch.to(device)
            x, y = batch, batch.y.float()
            y_hat = model(x)
            l = loss(y_hat, y)
            rps = right_preds_count(y_hat, y)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            stat = Stat(y_hat.tolist(), y.tolist(), l.item(), rps.item(), y.numel())
            train_stats(stat)
        history(train_stats, epoch + 1)

        if val_set_loader is not None:
            with torch.no_grad():
                model.eval()
                val_stats = Stats("validate set at epoch " + str(epoch + 1))
                for val_i, val_batch in enumerate(val_set_loader):
                    val_batch.to(device)
                    val_y_hat = model(val_batch)
                    val_l = loss(val_y_hat, val_batch.y.float())
                    val_rps = right_preds_count(val_y_hat, val_batch.y.float())
                    val_stat = Stat(val_y_hat.tolist(), val_batch.y.tolist(), val_l.item(), val_rps.item(), val_batch.y.numel())
                    val_stats(val_stat)
                history(val_stats, epoch + 1)
            # val_history.log()
        print(history)

# evaluate our model on test dataset
def eval(model, 
         test_set_loader,
         loss,
         device):
    print(f"evaluate model on test dataset...")
    with torch.no_grad():
        model.eval()
        test_stats = Stats("test process")
        for i, batch in enumerate(test_set_loader):
            batch.to(device)
            y_hat = model(batch)
            l = loss(y_hat, batch.y.float())
            rps = right_preds_count(y_hat, batch.y.float())
            stat = Stat(y_hat.tolist(), batch.y.tolist(), l.item(), rps.item(), batch.y.numel())
            test_stats(stat)
        #print(stats.outs(), stats.labels())
        metrics = Metrics(test_stats.outs(), test_stats.labels())
        # print(metrics)
        metrics.log()
    return metrics()

if __name__ == "__main__":
    all_config = Tensor_GGNN_GCN()
    
    model_config = all_config.model
    circle_ggnn_layer_config = model_config["gated_graph_conv_args"]

    conv_output_layer_config = model_config["conv_args"]
    data_embedding_size = model_config["emb_size"]

    # learning_rate = all_config.learning_rate
    # weight_decay = all_config.weight_decay
    # regular_lambda = all_config.loss_lambda
    l1_regular_lambda = 1e-6
    learning_rate = 5e-5
    weight_decay = 1.3e-6
    batch_size = 64
    epochs = 100

    # print(circle_ggnn_layer_config)
    # print(conv_output_layer_config)
    # print(data_embedding_size)

    run_device = try_device("second")
    
    graph_conv_output_layer_config = {"layer_num":5,
                                      "intra_gcn_paras":{"in_channels":101, "out_channels":101, "improved":False, "cached":False, "add_self_loops":False, "normalize":True, "bias":True},
                                      "inter_gcn_paras":{"in_channels":101, "out_channels":101, "improved":False, "cached":False, "add_self_loops":False, "normalize":True, "bias":True},
                                      "device":run_device}

    net = TensorGCNN(graph_conv_output_layer_config, conv_output_layer_config, out_size=101, emb_size=data_embedding_size)
    net.to(device=run_device)

    # total_parameters(net)
    # model_summary(net)
    
    loss_func = lambda o, t: F.binary_cross_entropy(o, t)# + F.l1_loss(o, t) * regular_lambda
    target_optim = O.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_set_loader, val_set_loader, test_set_loader = get_dataset_loader(batch_size=batch_size)
    # train_set_loader, val_set_loader, test_set_loader = get_multiclass_dataset_loader(batch_size, 
    #                                                                                   n_classes=1, 
    #                                                                                   class_select=["CWE-119", "CWE-120", "CWE-469", "CWE-476"], 
    #                                                                                   cache="Composite",
    #                                                                                   verbose=True)
    # train_set_loader, val_set_loader, test_set_loader = get469(6, batch_size, verbose=True)

    # test_(net, train_set_loader, run_device)

    train(net, 
          train_set_loader,
          val_set_loader, 
          epochs,
          target_optim,
          loss_func,
          run_device)
    
    model_save_path = "./saved_models/circle_ggnn_model"
    net.save(model_save_path)

    eval_metrics= eval(net, test_set_loader, loss_func, run_device)
    metrics = ["Accuracy", "Precision", "Recall", "F-measure"]
    for m in metrics:
        print(m, eval_metrics[m])