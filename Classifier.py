from matplotlib import pyplot as plt
import numpy as np
import os 
from pandas import read_csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd


def load_samples(filepath):
	Output = []  #list of 2d arrays
	dataframe = read_csv(filepath, header = None)
	Samples = dataframe.to_numpy() 
	Two_D = np.empty((0,Samples.shape[1]))

	for i in range(Samples.shape[0]):
		if not np.isnan(Samples[i,0]):   #if the first element of each line is not NaN
			Two_D = np.vstack([Two_D,Samples[i,:]])
		else:
			Output.append(Two_D) #stack on the 2d array
			Two_D = np.empty((0,Samples.shape[1]))   #empty the 2d array
			
	return Output #list of arrays

def create_dataset():
	#load data from csv
	__location__ = os.path.realpath(
		os.path.join(os.getcwd(), os.path.dirname(__file__)))

	Expert_Gestures = load_samples(os.path.join(__location__, 'ExpertSamples.csv'))
	Novice_Gestures = load_samples(os.path.join(__location__, 'NoviceSamples.csv'))
	#generate labels  1 = Expert 0 = Novice
	Combined_y = [1]*len(Expert_Gestures) + [0]*len(Novice_Gestures)

	#get max time step count
	max_steps = 0
	for i in range(len(Expert_Gestures)):
		if Expert_Gestures[i].shape[0] > max_steps:
			max_steps = Expert_Gestures[i].shape[0]
	for i in range(len(Novice_Gestures)):
		if Novice_Gestures[i].shape[0] > max_steps:
			max_steps = Novice_Gestures[i].shape[0]
				
	#pad all arrays to max step count
	for i in range(len(Expert_Gestures)):
		pad = ((0,max_steps-Expert_Gestures[i].shape[0]),(0,0))
		Expert_Gestures[i] = np.pad(Expert_Gestures[i],pad_width=pad,constant_values=0)
	for i in range(len(Novice_Gestures)):
		pad = ((0,max_steps-Novice_Gestures[i].shape[0]),(0,0))
		Novice_Gestures[i] = np.pad(Novice_Gestures[i],pad_width=pad,constant_values=0)

	#combine and stack into 3d array
	Combined_X = Expert_Gestures + Novice_Gestures
	Combined_X = np.stack(Combined_X)
	#Combined_X = np.moveaxis(Combined_X,0,2) 
	Combined_y = np.array(Combined_y)
	Combined_y = to_categorical(Combined_y)
	return Combined_X, Combined_y

def shuffle_and_split(Combined_X,Combined_y,ratio):
	#shuffle
	np.random.seed(np.random.randint(0,1000))
	np.random.shuffle(Combined_X)
	np.random.shuffle(Combined_y)
	#split
	train_percent = ratio
	test_percent = 1-train_percent
	train_X, test_X = np.split(Combined_X,[int(train_percent*Combined_X.shape[0])])
	train_y,test_y = np.split(Combined_y,[int(train_percent*Combined_y.shape[0])])
	return train_X,test_X,train_y,test_y

def train_model(model, optimizer, train, epochs=20):
	criterion = nn.MSELoss()
	for epoch in range(epochs):
		print("Epoch {}".format(epoch))
        y_true = list()
        y_pred = list()
        total_loss = 0
        for batch, targets, lengths, raw_data in create_dataset(train, x_to_ix, y_to_ix, bs=TRAIN_BATCH_SIZE):
            batch, targets, lengths = sort_batch(batch, targets, lengths)
            model.zero_grad()
            pred, loss = apply(model, criterion, batch, targets, lengths)
            loss.backward()
            optimizer.step()
            pred_idx = torch.max(pred, 1)[1]
            y_true += list(targets.int())
            y_pred += list(pred_idx.data.int())
            total_loss += loss
        acc = accuracy_score(y_true, y_pred)
        val_loss, val_acc = evaluate_validation_set(model, dev, x_to_ix, y_to_ix, criterion)
        print("Train loss: {} - acc: {} \nValidation loss: {} - acc: {}".format(list(total_loss.data.float())[0]/len(train), acc,
                                                                                val_loss, val_acc))
	return model


class LSTM(nn.Module):
	def __init__(self,input_dim,hidden_dim,label_dim):
		super(LSTM, self).__init__()	

		self.hidden_dim = hidden_dim
		self.input_dim = input_dim

		self.lstm = nn.LSTM(input_dim, hidden_dim)
		self.fully_connected = nn.Linear(hidden_dim, label_dim)
		self.softmax = nn.LogSoftmax()

	def init_hidden(self,batch_size):
		return(autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)), autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))


	def forward(self,batch):
		self.hidden = self.init_hidden(batch.size(2))
		output, (hidden_h,hidden_c) = self.lstm(batch,self.hidden)
		output = self.fully_connected(output)
		output = self.softmax(output)
		return output






if __name__ == '__main__':
	train_X,test_X,train_y,test_y = shuffle_and_split(create_dataset(),0.7)  #[time, feature, sample]
	net = LSTM(20,100,2,1)
	optimizer = optim.SGD(net.parameters(),lr = 0.01, weight_decay=1e-4)
	net = train_model(net,optimizer,train_X,train_y)

