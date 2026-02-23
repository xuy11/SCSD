from torch.utils.data import Dataset, DataLoader
import scipy.io
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import math


def loadMfDIMvMlDataFromMat(mat_path, fold_mat_path,fold_idx=0):
    data = scipy.io.loadmat(mat_path)
    datafold = scipy.io.loadmat(fold_mat_path)
    mv_data = data['X'][0]
    labels = data['label']
    labels = labels.astype(np.float32)
    if labels.min() == -1:
            labels = (labels + 1) * 0.5
    if labels.shape[0] in mv_data[0].shape:
        total_sample_num = labels.shape[0]
    elif labels.shape[1] in mv_data[0].shape:
        total_sample_num = labels.shape[0]
    if total_sample_num!=mv_data[0].shape[0]:
        mv_data = [v_data.T for v_data in mv_data]
    if total_sample_num!=labels.shape[0]:
        labels = labels.T
    assert mv_data[0].shape[0]==labels.shape[0]==total_sample_num

    folds_data = datafold['folds_data']
    folds_label = datafold['folds_label']
    folds_sample_index = datafold['folds_sample_index']
    inc_view_indicator = np.array(folds_data[0, fold_idx], 'int32')
    inc_label_indicator = np.array(folds_label[0, fold_idx], 'int32')
    sample_index = np.array(folds_sample_index[0, fold_idx], 'int32').reshape(-1)-1
    labels,inc_view_indicator,inc_label_indicator = labels[sample_index],inc_view_indicator[sample_index],inc_label_indicator[sample_index]
    mv_data = [v_data[sample_index,:] for v,v_data in enumerate(mv_data)]

    assert inc_view_indicator.shape[0]==inc_label_indicator.shape[0]==sample_index.shape[0]==labels.shape[0]
    inc_mv_data = [(StandardScaler().fit_transform(v_data.astype(np.float32))*inc_view_indicator[:,v:v+1]) for v,v_data in enumerate(mv_data)]

    inc_labels = labels*inc_label_indicator
    ind00 = labels.sum(axis=1)==0

    return inc_mv_data,inc_labels,labels,inc_view_indicator,inc_label_indicator,total_sample_num

class IncDataset(Dataset):
    def __init__(self,mat_path, fold_mat_path, training_ratio=0.7, val_ratio=0.15, fold_idx=0, mode='train',semisup=False):
        inc_mv_data, inc_labels, labels, inc_V_ind, inc_L_ind, total_sample_num= loadMfDIMvMlDataFromMat(mat_path,fold_mat_path,fold_idx)
        self.train_sample_num = math.ceil(total_sample_num * training_ratio)
        self.val_sample_num = math.ceil(total_sample_num * val_ratio)
        self.test_sample_num = total_sample_num - self.train_sample_num - self.val_sample_num
        if mode=='train':
            self.cur_mv_data = [v_data[:self.train_sample_num] for v_data in inc_mv_data]
            self.cur_labels = inc_labels[:self.train_sample_num]
            self.cur_inc_V_ind = inc_V_ind[:self.train_sample_num]
            self.cur_inc_L_ind = inc_L_ind[:self.train_sample_num]
        elif mode=='val':
            self.cur_mv_data = [v_data[self.train_sample_num:self.train_sample_num+self.val_sample_num] for v_data in inc_mv_data]
            self.cur_labels = labels[self.train_sample_num:self.train_sample_num+self.val_sample_num]
            self.cur_inc_V_ind = inc_V_ind[self.train_sample_num:self.train_sample_num+self.val_sample_num]
            self.cur_inc_L_ind = np.ones_like(inc_L_ind[self.train_sample_num:self.train_sample_num+self.val_sample_num])
        else:
            self.cur_mv_data = [v_data[self.train_sample_num+self.val_sample_num:] for v_data in inc_mv_data]
            self.cur_labels = labels[self.train_sample_num+self.val_sample_num:]
            self.cur_inc_V_ind = inc_V_ind[self.train_sample_num+self.val_sample_num:]
            self.cur_inc_L_ind = np.ones_like(inc_L_ind[self.train_sample_num+self.val_sample_num:])

        self.mode = mode
        self.classes_num = labels.shape[1]
        self.d_list = [da.shape[1] for da in inc_mv_data]

    def __len__(self):
        if self.mode == 'train':
            return self.train_sample_num 
        elif self.mode == 'val':
            return self.val_sample_num 
        else: return self.test_sample_num 
    
    def __getitem__(self, index):
        data = [torch.tensor(v[index],dtype=torch.float) for v in self.cur_mv_data]
        label = torch.tensor(self.cur_labels[index], dtype=torch.float)
        inc_V_ind = torch.tensor(self.cur_inc_V_ind[index], dtype=torch.int32)
        inc_L_ind = torch.tensor(self.cur_inc_L_ind[index], dtype=torch.int32)
        return data,label,inc_V_ind,inc_L_ind

def getIncDataloader(matdata_path, fold_matdata_path, training_ratio=0.7, val_ratio=0.15, fold_idx=0, mode='train',batch_size=1,num_workers=1,shuffle=False):
    dataset = IncDataset(matdata_path, fold_matdata_path, training_ratio=training_ratio, val_ratio=val_ratio, mode=mode, fold_idx=fold_idx)
    dataloder = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    return dataloder,dataset
    
if __name__=='__main__':
    dataloder,dataset = getIncDataloader('/disk/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view.mat','/disk/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view_MaskRatios_0.5_LabelMaskRatio_0.5_TraindataRatio_0.7.mat',training_ratio=0.7,mode='train',batch_size=3,num_workers=2)
    print(dataset.test_sample_num)
    labels = torch.tensor(dataset.cur_labels).float()