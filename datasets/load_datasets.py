from torch.utils.data import Dataset
import pandas as pd
import h5py, os
import numpy as np
import torch

class h5file_Dataset(Dataset):
    def __init__(self, csv_file, h5file_dir, chief_feature_dir, gigapath_feature_dir, datatype):
        self.csv_file = pd.read_csv(csv_file)
        self.h5file_dir = h5file_dir
        self.chief_feature_dir = chief_feature_dir
        self.gigapath_feature_dir = gigapath_feature_dir
        self.datatype = datatype
        if self.datatype == 'train':
            self.csv_index = [self.csv_file.columns.get_loc('train'),self.csv_file.columns.get_loc('train_label')]
            self.csv_file = self.csv_file[self.csv_file['train'].notna()].reset_index(drop=True)
            self.lenth = self.csv_file['train'].count()
        elif self.datatype == 'val':
            self.csv_index = [self.csv_file.columns.get_loc('val'),self.csv_file.columns.get_loc('val_label')]
            self.csv_file = self.csv_file[self.csv_file['val'].notna()].reset_index(drop=True)
            self.lenth = self.csv_file['val'].count()
        elif self.datatype == 'test':
            self.csv_index = [self.csv_file.columns.get_loc('test'),self.csv_file.columns.get_loc('test_label')]
            self.csv_file = self.csv_file[self.csv_file['test'].notna()].reset_index(drop=True)
            self.lenth = self.csv_file['test'].count()

    def __len__(self):
        return self.lenth
    
    def __getitem__(self, index):
        coods_path = os.path.join(self.h5file_dir, self.csv_file.iloc[index, self.csv_index[0]].replace('.pt', '.h5'))
        chief_feature_path = os.path.join(self.chief_feature_dir, self.csv_file.iloc[index, self.csv_index[0]])
        gigapath_feature_path = os.path.join(self.gigapath_feature_dir, self.csv_file.iloc[index, self.csv_index[0]])
        with h5py.File(coods_path, 'r') as coods_file:
            structured_coords = coods_file['coords'][:]
            coords = np.stack([structured_coords['x'], structured_coords['y']], axis=1).astype(np.float32)
        chief_features = torch.load(chief_feature_path)
        gigapath_features = torch.load(gigapath_feature_path)
        label = self.csv_file.iloc[index, self.csv_index[1]]
        return coords, chief_features, gigapath_features, label


 
