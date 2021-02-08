from matplotlib import pyplot as plt
import numpy as np
import os
from numpy.core.fromnumeric import mean
from numpy.lib.polynomial import poly
import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from tqdm import trange
import statistics as st

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, output_dim=12, p=0):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fully_connected = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=p)

    def init_hidden(self, batch_size, layers):  
        return(autograd.Variable(torch.randn(layers, batch_size, self.hidden_dim)), autograd.Variable(torch.randn(layers, batch_size, self.hidden_dim)))

    def forward(self, batch):
        self.hidden = self.init_hidden(batch.size(0), 1)
        hidden, last_hidden = self.lstm(batch.float(), self.hidden)
        hidden = torch.Tensor(hidden)
        pred = self.fully_connected(hidden[:,-3:,:])
        return pred


def train_model(model, dataloader, epochs=100):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    torch.autograd.set_detect_anomaly(True)

    train_accs = list()
    test_accs = list()

    for epoch in range(epochs):
        # feed entire batch all at once
        for i_batch, batch in enumerate(dataloader):
            model.zero_grad()
            pred = model.forward(batch['X'])
            loss = criterion(pred, batch['y'].float())
            loss.backward()
            optimizer.step()
            print("Epoch {} ".format(epoch) + "loss: {}".format(loss.item()))

        # train_accs.append(model_accuracy(model, train_dataset[:]['X'], train_dataset[:]['y'], False))
        # test_accs.append(model_accuracy(model, test_dataset[:]['X'], test_dataset[:]['y'], True))

    # print("End Of Training")
    return model #, train_accs, test_accs

def predict(model,initial,window=40,future=60):
    current = initial
    predicted = torch.empty((0, initial.size(1)))
    for i in range(int(future/3)):
        input = current[-window:,:]  # seq, features
        new = model.forward(input.unsqueeze(0))
        current = torch.cat((current, new.squeeze(0)),0)
        predicted = torch.cat((predicted, new.squeeze(0)),0)
    return predicted