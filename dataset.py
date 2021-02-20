from torch.utils.data import Dataset, DataLoader
import torch
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
from pandas import read_csv


class JigsawsDataset(Dataset):
    def __init__(self, csv_list):       
        self.X, self.y = self.generate_tensors(csv_list)

    def generate_tensors(self, csv_list):
        """
        input: names of csvs
        output: 3d array of dim (sample, sequence, feature)
        """
        # load data from csv
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        Gestures = list()
        Combined_y = list()
        for i in range(len(csv_list)):
            Gestures.append(self.load_samples(os.path.join(__location__, csv_list[i])))
            # generate labels 
            Combined_y = Combined_y + [i]*len(Gestures[i]) 

        # get max sequence length
        max_steps = 0
        for i in range(len(Gestures)):
            for j in range(len(Gestures[i])):
                if Gestures[i][j].shape[0] > max_steps:
                    max_steps = Gestures[i][j].shape[0]

        # pad all arrays to max step count
        for i in range(len(Gestures)):
            for j in range(len(Gestures[i])):
                pad = ((max_steps-Gestures[i][j].shape[0],0),(0,0))
                Gestures[i][j] = np.pad(Gestures[i][j],pad_width=pad,constant_values=0)

        Combined_X = list()
        # combine and stack into 3d X array & 2d y array
        for i in range(len(Gestures)):
            Combined_X = Combined_X + Gestures[i]
        Combined_X = np.stack(Combined_X)
        Combined_y = np.array(Combined_y)
        Combined_y = to_categorical(Combined_y)

        return [torch.from_numpy(Combined_X.astype(np.double)), torch.from_numpy(Combined_y.astype(np.double))]

    def load_samples(self, filepath):
        """
        input: path to csv
        output: list of 2d arrays of shape (sequence, feature)
        """
        Output = []
        dataframe = read_csv(filepath, header=None)
        Samples = dataframe.to_numpy()
        Two_D = np.empty((0, Samples.shape[1]))
        for i in range(Samples.shape[0]):
            if not np.isnan(Samples[i, 0]):  # if the first element of each line is not NaN
                Two_D = np.vstack([Two_D, Samples[i, :]])
            else:
                Output.append(Two_D)  # stack on the 2d array into 3d array
                Two_D = np.empty((0, Samples.shape[1])) 
        return Output

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'X': self.X[idx, :, :], 'y': self.y[idx, :]}
        return sample


class PredictionDataset(Dataset):
    def __init__(self, tensor, window, stride):       
        self.X, self.y = self.generate_sequences(tensor, window, stride)
        self.y = torch.squeeze(self.y)

    def generate_sequences(self, tensor, window, stride):
        tensor = torch.tensor(tensor.clone().detach())
        print(tensor.size())
        print(tensor)
        tensor.unsqueeze(0)
        print(tensor.size())

        X = list()
        y = list()
        length = tensor.size(1)
        for i in range(tensor.size(0)):
            for j in range(length-window-stride-1):
                if tensor[i,j,:].float().sum().item() != 0:  #only do the non-zero portions
                    train = tensor[i, j:j+window, :]              
                    label = tensor[i, j+window:j+window+stride, :]
                    X.append(train)
                    y.append(label)
        X = np.stack(X)
        y = np.stack(y)
        return [torch.from_numpy(X.astype(np.double)), torch.from_numpy(y.astype(np.double))]

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'X': self.X[idx,:,:], 'y': self.y[idx,:]}
        return sample
      