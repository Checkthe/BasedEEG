import argparse
import os.path as osp
import numpy as np
import torch

class cfg():
    def __init__(self):
        self.this_dir = osp.dirname(__file__)
        self.data_root = osp.abspath(osp.join(self.this_dir,'dataset'))

    def get_args(self):
        parser = argparse.ArgumentParser()
        #base
        parser.add_argument('--device',default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        parser.add_argument('--exp_id',default=2 ,help='0：只取刺激条件 1：只取中性条件 2：只取健康人群 3：只取成瘾人群')
        #data
        parser.add_argument('--if_load_def',default=True)
        parser.add_argument('--if_zscore',default=True)
        parser.add_argument('--if_bandpass',default=True)
        parser.add_argument('--if_filter',default=True)
        #path
        parser.add_argument('--BCI1',default=osp.abspath(osp.join(self.data_root,'BCI1\\Competition_train.mat')))
        parser.add_argument('--BCICIV2A',
                            default=[osp.abspath(osp.join(self.data_root, 'BCICIV_2a_gdf\\A0' + str(i + 1) + 'T.gdf')) for i in
                                     range(9)])
        parser.add_argument('--BCICIV2B',
                            default=[[osp.abspath(osp.join(self.data_root, 'BCICIV_2b_gdf\\B0'+str(j+1)+'0' + str(i + 1) + 'T.gdf'))
                                     for i in
                                     range(3)] for j in range(9)])
        parser.add_argument('--B2A',
                            default=[osp.abspath(osp.join(self.data_root, '2A_NPZ\\A0' + str(i + 1) + 'T.npz'))
                                     for i in
                                     range(9)])
        parser.add_argument('--B2B',
                            default=[osp.abspath(osp.join(self.data_root, '2B_NPZ\\B0' + str(i + 1) + 'T.npz'))
                                     for i in
                                     range(9)])
        parser.add_argument('--sleep_edfx',
                            default=[(osp.abspath(osp.join(self.data_root, 'Sleep-EDFx\\SC400'+str(i+1)+'E0-PSG.edf')),
                                      osp.abspath(osp.join(self.data_root, 'Sleep-EDFx\\SC400'+str(i+1)+'EC-Hypnogram.edf'))) for i in range(2)])
        parser.add_argument('--KaggleERN_Data',
                            default=[osp.abspath(osp.join(self.data_root, 'KaggleERN\\train\\Data_S02_Sess0' + str(i + 1) + '.csv'))
                                     for i in
                                     range(1)])
        parser.add_argument('--KaggleERN_Labels',
                            default=osp.abspath(
                                osp.join(self.data_root, 'KaggleERN\\TrainLabels.csv')))
        parser.add_argument('--P300',
                            default=[(osp.abspath(osp.join(self.data_root, 'P300\\rc0' + str(i + 1) + '.edf')),
                                      osp.abspath(osp.join(self.data_root, 'P300\\rc0' + str(i + 1) + '.edf.event')))
                                     for i in range(6)])
        parser.add_argument('--hc_data',
                            default=[osp.abspath(osp.join(self.data_root,'tobacco\\hc\\c'+str(i+1)+'\\data.mat')) for i in
                                     range(20)])
        parser.add_argument('--hc_conds',
                            default=[osp.abspath(osp.join(self.data_root, 'tobacco\\hc\\c' + str(i + 1) + '\\label.mat')) for i in
                                     range(20)])
        parser.add_argument('--hs_data',
                            default=[osp.abspath(osp.join(self.data_root,'tobacco\\hs\\s'+str(i+1)+'\\data.mat')) for i in
                                     range(20)])
        parser.add_argument('--hs_conds',
                            default=[osp.abspath(osp.join(self.data_root, 'tobacco\\hs\\s' + str(i + 1) + '\\label.mat')) for i in
                                     range(20)])
        parser.add_argument('--hs',default=[osp.abspath(osp.join(self.data_root,
                                        'tobacco_csv\\hs\\s' + str(i + 1) + '.csv')) for i in range(20)])
        parser.add_argument('--hc', default=[osp.abspath(osp.join(self.data_root,
                                        'tobacco_csv\\hs\\c' + str(i + 1) + '.csv')) for i in range(20)])
        #------------feature_engineering-----------
        parser.add_argument('--save_dae',default=osp.abspath(osp.join(self.this_dir, 'checkpoint/dae.pt')))
        parser.add_argument('--feats',default=osp.abspath(osp.join(self.data_root,'features/features.csv')))
        parser.add_argument('--feats_syn', default=osp.abspath(osp.join(self.data_root, 'features/feats_syn.csv')))
        parser.add_argument('--feats_scaled', default=osp.abspath(osp.join(self.data_root, 'features/feats_scaled.csv')))

        self.cfg = parser.parse_args()
        return self.cfg

if __name__ == '__main__':
    cfgs = cfg().get_args()
    print(cfgs.hc_data)