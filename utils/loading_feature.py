import torch
import numpy as np
from scipy.io import loadmat
import pandas as pd
from config import cfg
from preprocess.pipeline import prep_pipeline
from sklearn.model_selection import train_test_split,KFold
from torch.utils.data import Dataset, DataLoader

class Dataset_binary(Dataset):
    def __init__(self,data,labels=None,if_labels=True):
        self.data = data
        self.labels = labels
        self.if_labels = if_labels
        #归一化
        # self.data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)

    def __len__(self):
        return len(self.data)
    def attr(self):
        return self
    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        if self.if_labels:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return data,label
        else:
            return data

def create_loader(data,labels,k=5,random_state=42):
    train_dataloader,test_dataloader = [],[]
    kfold = KFold(n_splits=k, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
        data_train, data_test, labels_train, labels_test = data[train_idx],data[val_idx],labels[train_idx],labels[val_idx]
        # 分组数据
        train_data = Dataset_binary(data_train, labels_train)
        test_data = Dataset_binary(data_test, labels_test)
        # 创建数据加载器
        train_dataloader.append(DataLoader(train_data, batch_size=8, shuffle=True, drop_last=True))
        test_dataloader.append(DataLoader(test_data, batch_size=1, shuffle=True, drop_last=True))
    print('====数据加载完毕====')
    return train_dataloader,test_dataloader

class Dataset_DAE(Dataset):
    def __init__(self,data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def attr(self):
        return self
    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        return data

def load_dae():
    cfgs = cfg().get_args()
    data = loadmat(cfgs.hs_data[0])['data']
    # 滤波
    sf = 250
    bp_filter_range = (1, 40)
    data = prep_pipeline(data, sf=sf, fr=bp_filter_range)
    train_ds = Dataset_DAE(data)
    return train_ds

def load_singal():
    # 读取数据
    cfgs = cfg().get_args()
    dataset = pd.read_csv(cfgs.feats_scaled).iloc[:, 1:]
    feats = dataset.drop(columns=['labels']).to_numpy()
    labels = dataset['labels'].values
    train_dataloader, test_dataloader = create_loader(feats, labels)
    return train_dataloader,test_dataloader,(feats,labels)
