from curses import noecho
from pandas import DataFrame
from os.path import *
from os import *
import pandas as pd
from sklearn.metrics import davies_bouldin_score
from sklearn.model_selection import train_test_split
from torch import native_group_norm, negative
import torch
from torch_sparse import to_scipy

from input_dataset import InputDataset

dataset_basepath = "/home/passwd123/wch/VDoTR/dataset"
ds_dir = "/home/passwd123/wch/v_dataset/"
ds_cache_dir = "/home/passwd123/wch/c_dataset"

def load_dataset(file_path):
    dataset = pd.read_pickle(file_path)
    # dataset.info(memory_usage='deep')
    return dataset

def train_val_test_split(data_frame: pd.DataFrame, shuffle=True):
    print("Splitting Dataset")

    false = data_frame[data_frame.target == 0]
    true = data_frame[data_frame.target == 1]

    train_false, test_false = train_test_split(false, test_size=0.2, shuffle=shuffle)
    test_false, val_false = train_test_split(test_false, test_size=0.5, shuffle=shuffle)
    train_true, test_true = train_test_split(true, test_size=0.2, shuffle=shuffle)
    test_true, val_true = train_test_split(test_true, test_size=0.5, shuffle=shuffle)

    train = train_false.append(train_true)
    val = val_false.append(val_true)
    test = test_false.append(test_true)

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return InputDataset(train), InputDataset(test), InputDataset(val)

def get_dataset_loader(batch_size):
    dataset = DataFrame(columns=['input','target'])

    data_sets_files = sorted([f for f in listdir(dataset_basepath) if isfile(join(dataset_basepath, f))])

    dataset = DataFrame(columns=['input','target'])
    positive_samples = DataFrame(columns=['input','target'])
    for file in data_sets_files:
        # if file.startswith("cwe119"):
        #     dataset_sub = load_dataset(join(dataset_basepath, file))
        #     positive_samples = positive_samples.append(dataset_sub[dataset_sub.target == 1])
        #     if len(dataset[dataset.target == 0]) < len(dataset_sub[dataset_sub.target == 0]):
        #         dataset = dataset_sub[dataset_sub.target == 0]
        if file.startswith("cwe469"):
            positive_samples = load_dataset(join(dataset_basepath, file))
    dataset = dataset.append(positive_samples)

    dataset.info(memory_usage='deep')

    train_loader, val_loader, test_loader = list(
            map(
                lambda x: x.get_loader(batch_size, shuffle=True), 
                train_val_test_split(dataset, shuffle=True)
                )
            )
    
    return train_loader, val_loader, test_loader

def split_singleclass_dataset(df:pd.DataFrame):
    train, test = train_test_split(df, test_size=0.2, shuffle=True)
    test, val = train_test_split(test, test_size=0.5, shuffle=True)

    return train, val, test

def get_loader(train_ds, val_ds, test_ds, batch_size):
    train_loader, val_loader, test_loader = list(
            map(
                lambda x: x.get_loader(batch_size, shuffle=True), 
                [InputDataset(train_ds), InputDataset(val_ds), InputDataset(test_ds)]
                )
            )
    return train_loader, val_loader, test_loader

def get_single_all(ds_files, dp, verbose=False):
    single = DataFrame(columns=['input','target'])
    for ds_fn in filter(lambda x : x.startswith(dp), ds_files):
        single = single.append(load_dataset(join(ds_dir, ds_fn)))
    if verbose:
        print(f"{dp} counts up to {len(single)}.")
    train_single, val_single, test_single = split_singleclass_dataset(single)
    return train_single, val_single, test_single

def get_multiclass_dataset_loader(batch_size, 
                                  n_classes,
                                  class_select=None,
                                  cache=None,
                                  verbose=False,
                                  
                                  ):
    ds_suffix = ["_train", "_val", "_test"]
    if cache is not None:
        if exists(join(ds_cache_dir, cache)):
            train_all, val_all, test_all = [load_dataset(join(ds_cache_dir, cache + s)) for s in ds_suffix]
            return get_loader(train_all, val_all, test_all, batch_size)
        else:
            mknod(join(ds_cache_dir, cache))

    ds_files = [f for f in listdir(ds_dir) if isfile(join(ds_dir, f))]
    ds_prefix = ["Non_vulnerable", "CWE-119", "CWE-120", "CWE-469", "CWE-476", "CWE-other"]
    train_all, val_all, test_all = DataFrame(columns=['input','target']), DataFrame(columns=['input','target']), DataFrame(columns=['input','target'])

    if n_classes == 1:
        assert isinstance(class_select, list)
        ds_select = [ds_prefix[0]] + class_select
    else:
        ds_select = ds_prefix[0:n_classes]

    for dp in ds_select:
        train_single, val_single, test_single = get_single_all(ds_files, dp, verbose=verbose)
        train_all = train_all.append(train_single)
        val_all = val_all.append(val_single)
        test_all = test_all.append(test_single)
    
    train_all = train_all.reset_index(drop=True)
    val_all = val_all.reset_index(drop=True)
    test_all = test_all.reset_index(drop=True)
    
    if n_classes == 1:
        la = lambda x : 1 if x > 0 else 0

        train_all["target"] = train_all["target"].apply(lambda x : torch.tensor([0]) if torch.argmax(x) == 0 else torch.tensor([1]))
        val_all["target"] = val_all["target"].apply(lambda x : torch.tensor([0]) if torch.argmax(x) == 0 else torch.tensor([1]))
        test_all["target"] = test_all["target"].apply(lambda x : torch.tensor([0]) if torch.argmax(x) == 0 else torch.tensor([1]))

        for d in train_all["input"]:
            d.y = la(d.y)
        for d in val_all["input"]:
            d.y = la(d.y)
        for d in test_all["input"]:
            d.y = la(d.y)

    if cache is not None:
        t = (train_all, val_all, test_all)
        for i in range(len(ds_suffix)):
            pd.to_pickle(t[i], join(ds_cache_dir, cache + ds_suffix[i]))

    if verbose:
        print("train set info: ")
        train_all.info(memory_usage='deep')
        print("======================split line=========================")
        print("val set info: ")
        val_all.info(memory_usage='deep')
        print("======================split line=========================")
        print("test set info: ")
        test_all.info(memory_usage='deep')
        print("======================split line=========================")

    # return train_all, val_all, test_all
    return get_loader(train_all, val_all, test_all, batch_size)

def get469(multiple, batch_size, verbose=False):
    ds_files = [f for f in listdir(ds_dir) if isfile(join(ds_dir, f))]

    nv_single = DataFrame(columns=['input','target'])
    for ds_fn in filter(lambda x : x.startswith("Non_vulnerable"), ds_files):
        nv_single = nv_single.append(load_dataset(join(ds_dir, ds_fn)))
    
    v469_single = DataFrame(columns=['input','target'])
    for ds_fn in filter(lambda x : x.startswith("CWE-469"), ds_files):
        v469_single = v469_single.append(load_dataset(join(ds_dir, ds_fn)))
    
    nv_single = nv_single[0:multiple * len(v469_single)]

    if verbose:
        print(f"Non_vulnerable count up to {len(nv_single)}")
        print(f"CWE-469 count up to {len(v469_single)}")
        v469_single.info()
        nv_single.info()

    nv_train, nv_val, nv_test = split_singleclass_dataset(nv_single)
    v469_train, v469_val, v469_test = split_singleclass_dataset(v469_single)

    all_train = v469_train.append(nv_train)
    all_val = v469_val.append(nv_val)
    all_test = v469_test.append(nv_test)

    la = lambda x : 1 if x > 0 else 0
    for d in all_train["input"]:
        d.y = la(d.y)
    for d in all_val["input"]:
        d.y = la(d.y)
    for d in all_test["input"]:
        d.y = la(d.y)

    return get_loader(all_train, all_val, all_test, batch_size)

def to_disk():
    get_multiclass_dataset_loader(32, n_classes=1, class_select=["CWE-119"], cache="CWE-119", verbose=False)
    get_multiclass_dataset_loader(32, n_classes=1, class_select=["CWE-120"], cache="CWE-120", verbose=False)
    # get_multiclass_dataset_loader(32, n_classes=1, class_select=["CWE-469"], cache="CWE-469", verbose=False)
    get_multiclass_dataset_loader(32, n_classes=1, class_select=["CWE-476"], cache="CWE-476", verbose=False)
    get_multiclass_dataset_loader(32, n_classes=1, class_select=["CWE-119", "CWE-120", "CWE-469", "CWE-476"], cache="Composite", verbose=False)

if __name__ == "__main__":
    # ds_frame, _, _, ds_loader = get_dataset_loader(2)
    # for i in range(2):
    #     print(ds_frame.iloc[i].input.cfg_edge_index)
    #     print(ds_frame.iloc[i].input.ast_edge_index)
    #     print(ds_frame.iloc[i].input.ddg_edge_index)
    #     print(ds_frame.iloc[i].input.ncs_edge_index)
    # print("=================================================")
    # print(isinstance(ds_frame.iloc[63].input, torch_geometric.data.Data))    # Data
    # for i, batch in enumerate(ds_loader):
    #     if i == 0:
    #         x, y = batch, batch.y.float()
    #         print(batch.cfg_edge_index)
    #         print(batch.ast_edge_index)
    #         print(batch.ddg_edge_index)
    #         print(batch.ncs_edge_index)

    # train, val, test = get_multiclass_dataset_loader(32, 1, verbose=False, class_select="CWE-120")
    # print(type(train["input"].iloc[0]))
    # print(type(train["target"].iloc[0]))
    # print([x.y for x in train["input"][1:5]])

    # train["target"] = train["target"].apply(lambda x : torch.argmax(x))
    # print(train["input"][len(train)-3].y)

    to_disk()