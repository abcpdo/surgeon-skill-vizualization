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
    def __init__(self, input_dim, hidden_dim, layers=1, label_dim=2, p=0):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, batch_first=True)
        self.fully_connected = nn.Linear(hidden_dim, label_dim)
        # self.logsoftmax = nn.LogSoftmax()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=p)

    def init_hidden(self, batch_size, layers):   #init as (batch_size, timesteps, hidden_dim)
        return(autograd.Variable(torch.randn(layers, batch_size, self.hidden_dim)), autograd.Variable(torch.randn(layers, batch_size, self.hidden_dim)))

    def forward(self, batch, layers):
        self.hidden = self.init_hidden(batch.size(0), layers)
        hidden, last_hidden = self.lstm(batch.float(), self.hidden)
        output = self.dropout(last_hidden[0].squeeze())
        output = self.fully_connected(output)
        return output


def train_model(model, train_dataloader, test_dataloader, train_dataset, test_dataset, epochs=20):
    #train_dataloader
    #pos_weight = torch.tensor([(train_y[:, 1] == 1).sum()/(train_y[:, 0] == 1).sum(), (train_y[:, 0] == 1).sum()/(train_y[:, 1] == 1).sum()])     # negative/positive of expert/novice class for pos_weight
    # print(pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    # print("Pos_Weight:")
    # print(pos_weight)
    criterion = nn.BCEWithLogitsLoss()
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


def model_accuracy(model, X, y,Test_flag):
    model.eval()
    pred = model.forward(X.float(), 1)
    predicted = torch.max(pred, 1)[1]
    actual = torch.max(y.float(), 1)[1]
    correct = ((predicted == actual) == True).sum()
    total = X.size(0)

    # if Test_flag:
    # 	print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))
    # else:
    # 	print('Accuracy of the network on the train set: %d %%' % (100 * correct / total))

    model.train()
    return (correct/total)*100


if __name__ == '__main__':
    epochs = 400
    reshuffle = 5
    hidden_dim = 30

    accs = list() #list of final accuracies
    all_accs = list()

    for i in trange(reshuffle):  #reshuffle each loop
        #prepare data
        Combined_X, Combined_y = create_dataset('ExpertSamplesG4.csv','NoviceSamplesG4.csv')
        train_X,test_X,train_y,test_y = shuffle_and_split(Combined_X, Combined_y,0.7,i+55)
        # print("\nTrain Shape:")
        print(train_X.size())

        #train and evaluate model
        model = LSTMClassifier(train_X.size(2),hidden_dim,1,2,0) #input dim, hidden dim, num_layers, output dim, dropout ratio
        model,train_accs,test_accs = train_model(model,train_X,train_y,epochs) # model, X, y, epochs
        acc = model_accuracy(model,test_X[:,100:200,:],test_y,True)
        accs.append(acc.item())
        all_accs.append([train_accs,test_accs])

        # #display number of params
        # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        # params = sum([np.prod(p.size()) for p in model_parameters])
        # print(params)
    print(accs)
    print("Mean: {}".format(st.mean(accs)))
    # print("stddev: {}".format(st.stdev(accs)))

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    all_accs = np.array(all_accs)
    plt.plot(np.transpose(np.mean(all_accs[:,0],axis = 0)),label='Train', color = (0.5+np.random.random()*0.5,np.random.random()*0.1,np.random.random()*0.1))
    plt.plot(np.transpose(np.mean(all_accs[:,1],axis = 0)),label='Test', color = (np.random.random()*0.1,0.5+np.random.random()*0.5,np.random.random()*0.1))
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

    # save trained model
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    PATH = __location__ + '/LSTM.pth'
    torch.save(model.state_dict(), PATH)
    PATH = __location__ + '/fig.png'
    plt.savefig(PATH,dpi=250)
