import numpy as np
import torch
from torch import nn
from modeling.ERNCNN import ERNCNN
from baseline.ResNeXt import resnext50_32x4d
from utils.evaluation import evaluate_model
from utils.loading import (loaddata_singal, loaddata_BCICIV2B, loaddata_BCICIV2A)
from config import cfg

def train(model,train_loader,epoch):
    valid_epoch = int(epoch / 2)
    acc, kappa, ba, wf = [], [], [], []
    for sn in range(epoch):
        model.train()
        total_loss = 0
        for eeg, labels, bp in train_loader:
            eeg, labels, bp = (eeg.to(device, non_blocking=True),
                                labels.to(device, non_blocking=True),
                                bp.to(device, non_blocking=True))
            optimizer.zero_grad()
            outputs = model(bp)
            Closs = criterion(outputs, labels)
            loss = Closs
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {sn + 1}, Loss: {total_loss / len(train_loader):.4f}")
        if sn > valid_epoch:
            metrics = evaluate_model(model,test_loader=test_loader,detailed_report=False,plot_results=False)
            acc.append(metrics['accuracy'])
            kappa.append(metrics['cohen_kappa'])
            ba.append(metrics['balanced_accuracy'])
            wf.append(metrics['weighted_f1'])
            del metrics
    max_idx = np.argmax(acc)
    acc_max,kappa_max = acc[max_idx],kappa[max_idx]
    ba_max,wf_max = ba[max_idx],wf[max_idx]
    print("=== 本轮评估结果 ===")
    print(f"最优准确率 (Acc Best)   : {acc_max:.4f}")
    print(f"最优Kappa值 (Kappa Best): {kappa_max:.4f}")
    print(f"最优BA值 (BA Best)   : {ba_max:.4f}")
    print(f"最优WF1值 (WF1 Best): {wf_max:.4f}")
    return acc_max,kappa_max,ba_max,wf_max

if __name__ == '__main__':
    # 默认指定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载数据
    cfgs = cfg().get_args()
    train_dataloader, test_dataloader, dataset = loaddata_singal()
    # 模型参数
    n_trials = dataset[0].shape[0]
    n_channels = dataset[0].shape[1]
    n_points = dataset[0].shape[2]
    n_classes = len(np.unique(train_dataloader[0].dataset.labels))
    print(f"样本量 : {n_trials}")
    print(f"通道数 n_channels: {n_channels}")
    print(f"每段点数 n_points: {n_points}")
    #交叉验证
    model_hub = []
    accmax_hub,kappamax_hub = [],[]
    ba_hub, wf_hub = [],[]
    for fold,train_loader in enumerate(train_dataloader):
        test_loader = test_dataloader[fold]
        # 创建模型
        model = ERNCNN(n_classes=n_classes,in_chans=n_channels,n_points=n_points)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=0.01)
        # 统计参数量
        # total_params = sum(p.numel() for p in model.parameters())
        # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print(f"Total parameters: {total_params:,}")
        # print(f"Trainable parameters: {trainable_params:,}")
        #训练
        epoch = 300
        acc_max,kappa_max,ba_max,wf_max = train(model,train_loader,epoch)
        accmax_hub.append(acc_max)
        kappamax_hub.append(kappa_max)
        ba_hub.append(ba_max)
        wf_hub.append(wf_max)
        del model, optimizer, criterion
    best_idx = np.argmax(accmax_hub)
    acc_best,kappa_best,ba_best,wf_best = accmax_hub[best_idx],kappamax_hub[best_idx],ba_hub[best_idx],wf_hub[best_idx]
    acc_mean, kappa_mean, ba_mean, wf_mean = np.sum(accmax_hub)/5, np.sum(kappamax_hub)/5, np.sum(ba_hub)/5, np.sum(wf_hub)/5
    print("=== 模型评估结果 ===")
    print(f"最优准确率 (Acc Best)   : {acc_best:.4f}")
    print(f"最优Kappa值 (Kappa Best): {kappa_best:.4f}")
    print(f"最优BA值 (BA Best): {ba_best:.4f}")
    print(f"最优WF值 (WF Best): {wf_best:.4f}")
    print(f"平均准确率 (Acc Mean)   : {acc_mean:.4f}")
    print(f"平均Kappa值 (Kappa Mean): {kappa_mean:.4f}")
    print(f"平均BA值 (BA Mean)   : {ba_mean:.4f}")
    print(f"平均WF值 (WF Mean)   : {wf_mean:.4f}")
    print(accmax_hub)


