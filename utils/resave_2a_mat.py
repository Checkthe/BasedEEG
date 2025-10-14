import numpy as np
from scipy.io import loadmat

paths = [f'D:\machine_learning\ConvEEG\dataset\\2A_MAT\A0{idx+1}T.mat' for idx in range(9)]
save_paths = [f'D:\machine_learning\ConvEEG\dataset\\2A_NPZ\A0{idx+1}T.npz' for idx in range(9)]

for pidx,path in enumerate(paths):
    dataset = loadmat(path)
    data = dataset['data'][0]
    ip = len(data)-6
    trials = []
    labels = []
    for idx in range(6):
        idx = idx+ip
        struct = data[idx]
        x = struct['X'][0][0].T
        y = struct['y'][0][0].squeeze(1)
        trial = struct['trial'][0][0].squeeze(1)
        rows = []
        for t in trial:
            row = x[:,t:t+1800]
            rows.append(row)
        rows = np.stack(rows,axis=0)
        labels.append(y)
        trials.append(rows)
    trials = np.concatenate(trials,axis=0)
    labels = np.concatenate(labels,axis=0)
    labels[labels == 1] = 0
    labels[labels == 2] = 1
    labels[labels == 3] = 2
    labels[labels == 4] = 3
    np.savez(save_paths[pidx],data=trials,labels=labels)

