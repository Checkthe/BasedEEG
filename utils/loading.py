import torch
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import KFold
from config import cfg
from torch.utils.data import Dataset, DataLoader
from utils.hilbert_transform import hilbert_band_power
from utils.process_gdf import read_gdf_2b,read_gdf_2a
from preprocess.pipeline import prep_pipeline
from feature_engineering.channel_config import index_singal

class Dataset_binary(Dataset):
    def __init__(self,data,labels):
        self.cfgs = cfg().get_args()
        #Hilbert
        bps,names = hilbert_band_power(data)
        self.bps = (bps- np.mean(bps, axis=1, keepdims=True)) / np.std(bps, axis=1, keepdims=True)

        self.data = (data - np.mean(data, axis=2, keepdims=True)) / np.std(data, axis=2, keepdims=True)
        self.labels = labels.reshape(-1)

    def __len__(self):
        return len(self.data)
    def attr(self):
        return self
    def __getitem__(self, idx):
        eeg = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        bp = torch.tensor(self.bps[idx], dtype=torch.float32)
        return eeg,label,bp

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
def loaddata_singal(crt=True):
    cfgs = cfg().get_args()
    data = loadmat(cfgs.hs_data[0])['data'][:,index_singal,:]
    labels = loadmat(cfgs.hs_conds[0])['label'].squeeze(1)
    # 滤波
    sf = 250
    bp_filter_range = (1,40)
    bs_filter_range = (6,11)
    data = prep_pipeline(data, sf=sf ,fr=bp_filter_range)
    dataset = (data, labels)
    if crt:
        train_dataloader, test_dataloader = create_loader(data,labels)
        return train_dataloader, test_dataloader, dataset
    else:
        return dataset

def loaddata_BCI(crt=True):
    cfgs = cfg().get_args()
    dataset = loadmat(cfgs.BCI1)
    data = dataset['X']
    labels = dataset['Y'].squeeze(-1)
    labels[labels==-1] = 0
    data = data.astype(data.dtype.newbyteorder('='))
    # 滤波
    fs = 250
    bp_filter_range = (0.1, 30)
    dataset = (data, labels)
    if crt:
        train_dataloader, test_dataloader = create_loader(data, labels)
        return train_dataloader, test_dataloader, dataset
    else:
        return dataset

def loaddata_BCICIV2B(crt=True):
    cfgs = cfg().get_args()
    path = cfgs.B2B[2]
    dataset = np.load(path)
    data = dataset['data']
    labels = dataset['labels']
    #滤波
    sf = 250
    bp_filter_range = (0.5, 45)  #alpha、beta波
    data = prep_pipeline(data, sf=sf, fr=bp_filter_range)
    dataset = (data, labels)
    if crt:
        train_dataloader, test_dataloader = create_loader(data, labels)
        return train_dataloader, test_dataloader, dataset
    else:
        return dataset

def loaddata_BCICIV2A(crt=True):
    cfgs = cfg().get_args()
    path = cfgs.B2A[0]
    dataset = np.load(path)
    data = dataset['data']
    labels = dataset['labels']
    #滤波
    sf = 250
    bp_filter_range = (0.5, 45)  #alpha、beta波
    data = prep_pipeline(data, sf=sf, fr=bp_filter_range)
    dataset = (data, labels)
    if crt:
        train_dataloader, test_dataloader = create_loader(data, labels)
        return train_dataloader, test_dataloader, dataset
    else:
        return dataset



