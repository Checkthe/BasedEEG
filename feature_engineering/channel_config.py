singal_electrodes = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
    'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
    'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'OZ', 'O2', 'HEOG', 'VEOG'
]
channel_for_tobacco = ['FP1', 'FP2', 'AF3', 'AF4', 'F3', 'FZ', 'F4', 'FC3', 'FC4', 'C3',
                       'CZ', 'C4', 'CP3', 'CPZ', 'CP4', 'P3', 'PZ', 'P4', 'PO3', 'PO4', 'O2',
                       'P7', 'P8', 'T7', 'T8', 'TP7', 'TP8', 'PO7', 'OZ']

channel_for_tobacco = [
    'P3','P4','CPZ','CZ','PZ',  #事件相关，高判别
    'FP1', 'FP2', 'FZ',  #情绪相关，高判别
    'OZ', 'O2',  #视觉加工
    'PO7','P7','P8',
    'F3','F4',  #情绪相关
    'TP7','TP8'  #记忆联想
]
index_singal = []

for idx,item in enumerate(channel_for_tobacco):
    index_singal.append(singal_electrodes.index(item))


if __name__ == '__main__':
    print(index_singal)