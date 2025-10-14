from scipy.io import loadmat

dataset = loadmat('D:\machine_learning\ConvEEG\dataset\\2A_MAT\A01T.mat')
data = dataset['data']

print(data[0][3]['y'])

# import mne
# import matplotlib.pyplot as pl
# import numpy as np
# import os
#
# filename = "D:\machine_learning\ConvEEG\dataset\BCICIV_2a_gdf\A01E.gdf"  # 文件位置根据实际情况修改
# raw = mne.io.read_raw_gdf(filename)
# raw.plot()
# pl.show()
# # print(raw.info)
# # print(raw.ch_names)

