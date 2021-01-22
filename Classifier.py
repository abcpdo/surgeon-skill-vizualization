from matplotlib import pyplot as plt
import numpy as np
import os 
from pandas import read_csv
import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from tensorflow.keras.utils import to_categorical


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
	
	return torch.from_numpy(train_X.astype(np.double)),torch.from_numpy(test_X.astype(np.double)),torch.from_numpy(train_y.astype(np.double)), torch.from_numpy(test_y.astype(np.double))

class LSTM(nn.Module):
	def __init__(self,input_dim,hidden_dim,label_dim):
		super(LSTM, self).__init__()	
		self.hidden_dim = hidden_dim
		self.input_dim = input_dim
		self.label_dim = label_dim
		self.lstm = nn.LSTM(input_dim, hidden_dim)
		self.fully_connected = nn.Linear(hidden_dim, label_dim)
		self.softmax = nn.LogSoftmax()

	def init_hidden(self,batch_size):   #init as (1, timesteps, hidden_dim)
		return(autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)), autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))

	def forward(self,batch):
		self.hidden = self.init_hidden(batch.size(1))
		output, hidden = self.lstm(batch,self.hidden)
		output = self.fully_connected(output)
		output = self.softmax(output)
		return output

def train_model(model, train_X, train_y, epochs=20):
	optimizer = optim.SGD(model.parameters(),lr = 0.01, weight_decay=1e-4)
	criterion = nn.CrossEntropyLoss()

	for epoch in range(epochs):
		print(epoch)
		y_true = list()
		y_pred = list()
		total_loss = 0
		running_loss = 0
		model.zero_grad()
		pred = model.forward(torch.autograd.Variable(train_X))
		loss = criterion(pred,torch.autograd.Variable(train_y))
		loss.backward()
		optimizer.step()
		#print(pred)
		predicted = torch.max(pred, 1)[1]
		#print(predicted)
		y_true += list(targets.int())
		y_pred += list(predicted.data.int())
		total_loss += loss
		acc = accuracy_score(y_true, y_pred)
	return model

def test_model(model, test_X, test_y):
	model = model
	return 0


if __name__ == '__main__':
	Combined_X, Combined_y = create_dataset()
	train_X,test_X,train_y,test_y = shuffle_and_split(Combined_X, Combined_y,0.7)  #[sample, time, feature]
	net = LSTM(20,100,2)
	net = train_model(net,train_X.float(),train_y.long())
	net = test_model(net,test_X.float(),test_y.float())

