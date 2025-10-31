import os
import numpy as np
from torch.utils.data import Dataset

class Feature_Dataset(Dataset):
    def __init__(self,file_path,mode):
        self.path = file_path
        self.files_HiddenStates = []
        HiddenStates = os.path.join(file_path,"HiddenStates")
        for file in os.listdir(HiddenStates):
            data = np.load(os.path.join(HiddenStates,file))
            self.files_HiddenStates.append(data[0])
            
        Original_feature = os.path.join(file_path,"OriginalFeature")
        self.files_Original_feature = []
        for file in os.listdir(Original_feature):
            data = np.load(os.path.join(Original_feature,file))
            self.files_Original_feature.append(data[0])
            
        if mode == "train":
            self.files_HiddenStates = self.files_HiddenStates[:int(len(self.files_HiddenStates)*0.80)]
            self.files_Original_feature = self.files_Original_feature[:int(len(self.files_Original_feature)*0.80)]
        elif mode == "test":
            self.files_HiddenStates = self.files_HiddenStates[int(len(self.files_HiddenStates)*0.80):]
            self.files_Original_feature = self.files_Original_feature[int(len(self.files_Original_feature)*0.80):]
        else:
            raise ValueError("mode should be 'train' or 'test'")

    def __getitem__(self, index):
        HiddenStates = self.files_HiddenStates[index]
        Original_feature = self.files_Original_feature[index]
        return HiddenStates,Original_feature

    def __len__(self):
        return len(self.files_HiddenStates)