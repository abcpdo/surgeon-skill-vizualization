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

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=30, layers=1, label_dim=2, p=0):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, batch_first=True)
        self.fully_connected = nn.Linear(hidden_dim, label_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=p)

    def init_hidden(self, batch_size, layers): 
        return(autograd.Variable(torch.randn(layers, batch_size, self.hidden_dim)), autograd.Variable(torch.randn(layers, batch_size, self.hidden_dim)))

    def forward(self, batch, layers):
        self.hidden = self.init_hidden(batch.size(0), layers)
        hidden, last_hidden = self.lstm(batch.float(), self.hidden)
        output = self.dropout(last_hidden[0].squeeze())
        output = self.fully_connected(output)
        return output


def train_model(model, train_dataloader, test_dataloader, train_dataset, test_dataset, epochs=20):
    #train_dataloader
    pos_weight = torch.tensor([(train_dataset[:]['y'][:,1] == 1).sum()/(train_dataset[:]['y'][:,0] == 1).sum(), (train_dataset[:]['y'][:, 0] == 1).sum()/(train_dataset[:]['y'][:,1] == 1).sum()])     # negative/positive of expert/novice class for pos_weight
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    total_loss = 0
    torch.autograd.set_detect_anomaly(True)

    train_accs = list()
    test_accs = list()

    for epoch in trange(epochs):
        # feed entire batch all at once
        for i_batch, batch in enumerate(train_dataloader):
            model.zero_grad()
            # print(train_X[i].unsqueeze(0).shape)    #inserted dim of 1 to 0th dimension
            pred = model.forward(batch['X'], 1)
            # print(pred)
            # print(train_y[i].shape)
            loss = criterion(pred, batch['y'])
            loss.backward()
            optimizer.step()
            # print("Epoch {} ".format(epoch) + "loss: {}".format(loss.item()))
            total_loss += loss.item()

        train_accs.append(model_accuracy(model, train_dataset[:]['X'], train_dataset[:]['y'], False))
        test_accs.append(model_accuracy(model, test_dataset[:]['X'], test_dataset[:]['y'], True))

    # print("End Of Training")
    return model, train_accs, test_accs

def plot(end_accs,all_accs):
    print(end_accs)
    if len(end_accs) > 1:
        print("Stdev: {}".format(st.stdev(end_accs)))
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    all_accs = np.array(all_accs)
    plt.plot(np.transpose(np.mean(all_accs[:,0], axis = 0)),label='Train', color = (0.5+np.random.random()*0.5,np.random.random()*0.1,np.random.random()*0.1))
    plt.plot(np.transpose(np.mean(all_accs[:,1], axis = 0)),label='Test', color = (np.random.random()*0.1,0.5+np.random.random()*0.5,np.random.random()*0.1))
    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    plt.legend(newHandles, newLabels)
    plt.title('LSTM Expert/Novice Classifier')
    axis = plt.gca()
    axis.set_ylim(axis.get_ylim()[::-1])
    plt.show()

def model_accuracy(model, X, y, Test_flag):
    model.eval()
    pred = model.forward(X.float(), 1)
    Sigmoid = nn.Sigmoid()
    pred = Sigmoid(pred)
    predicted = torch.max(pred, 1)[1]
    actual = torch.max(y.float(), 1)[1]
    correct = ((predicted == actual) == True).sum()
    total = X.size(0)

    model.train()
    return (correct/total)*100
