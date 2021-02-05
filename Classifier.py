from matplotlib import pyplot as plt
import numpy as np
import os
from numpy.core.fromnumeric import mean
from numpy.lib.polynomial import poly
from pandas import read_csv
import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from tensorflow.keras.utils import to_categorical
from tqdm import trange
import statistics as st

def load_samples(filepath):
	"""
		input: path to csv
		output: list of 2d arrays of shape (sequence, feature)
	"""
	Output = []
	dataframe = read_csv(filepath, header = None)
	Samples = dataframe.to_numpy()
	Two_D = np.empty((0,Samples.shape[1]))

	for i in range(Samples.shape[0]):
		if not np.isnan(Samples[i,0]):   #if the first element of each line is not NaN
			Two_D = np.vstack([Two_D,Samples[i,:]])
		else:
			Output.append(Two_D) #stack on the 2d array
			Two_D = np.empty((0,Samples.shape[1]))   #empty the 2d array

	return Output

def create_dataset(name1 = 'ExpertSamplesG4.csv', name2 = 'NoviceSamplesG4.csv'):
	"""
		output: array of dim (sample, sequence, feature)
	"""
	#load data from csv
	__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
	Expert_Gestures = load_samples(os.path.join(__location__, name1))
	Novice_Gestures = load_samples(os.path.join(__location__, name2))
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
		pad = ((max_steps-Expert_Gestures[i].shape[0],0),(0,0))
		Expert_Gestures[i] = np.pad(Expert_Gestures[i],pad_width=pad,constant_values=0)
	for i in range(len(Novice_Gestures)):
		pad = ((max_steps-Novice_Gestures[i].shape[0],0),(0,0))
		Novice_Gestures[i] = np.pad(Novice_Gestures[i],pad_width=pad,constant_values=0)

	#combine and stack into 3d array
	Combined_X = Expert_Gestures + Novice_Gestures
	Combined_X = np.stack(Combined_X)
	Combined_y = np.array(Combined_y)
	Combined_y = to_categorical(Combined_y)
	return Combined_X, Combined_y

def shuffle_and_split(Combined_X,Combined_y,train_ratio,shuffle_index):
	"""
		input: 3d arrays, train/total ratio, shuffle index
		output: tensors of train_X, train_y, test_X, test_y
	"""
	#shuffle
	np.random.seed(shuffle_index)
	shuffle_array = np.arange(Combined_X.shape[0])
	np.random.shuffle(shuffle_array)
	Combined_X = Combined_X[shuffle_array]
	Combined_y = Combined_y[shuffle_array]

	#split
	test_ratio = 1-train_ratio
	train_X, test_X = np.split(Combined_X,[int(train_ratio*Combined_X.shape[0])])
	train_y,test_y = np.split(Combined_y,[int(train_ratio*Combined_y.shape[0])])

	return torch.from_numpy(train_X.astype(np.double)),torch.from_numpy(test_X.astype(np.double)),torch.from_numpy(train_y.astype(np.double)), torch.from_numpy(test_y.astype(np.double))


class Classifier(nn.Module):
	def __init__(self,input_dim,hidden_dim,layers,label_dim,p=0):
		super(Classifier, self).__init__()
		self.hidden_dim = hidden_dim
		self.input_dim = input_dim
		self.label_dim = label_dim
		self.lstm = nn.LSTM(input_dim, hidden_dim, layers,batch_first=True)
		self.fully_connected = nn.Linear(hidden_dim, label_dim)
		#self.logsoftmax = nn.LogSoftmax()
		self.sigmoid = nn.Sigmoid()
		self.dropout = nn.Dropout(p=p)


	def init_hidden(self,batch_size,layers):   #init as (batch_size, timesteps, hidden_dim)
		return(autograd.Variable(torch.randn(layers, batch_size, self.hidden_dim)), autograd.Variable(torch.randn(layers,batch_size, self.hidden_dim)))

	def forward(self,batch,layers):
		self.hidden = self.init_hidden(batch.size(0),layers)
		hidden, last_hidden = self.lstm(batch,self.hidden)
		output = self.dropout(last_hidden[0].squeeze())
		output = self.fully_connected(output)
		#output = self.sigmoid(output)

		return output


def train_model(model, train_X, train_y, epochs=20):
	pos_weight = torch.tensor([(train_y[:,1]==1).sum()/(train_y[:,0]==1).sum(),(train_y[:,0]==1).sum()/(train_y[:,1]==1).sum()])     #negative/positive of expert/novice class for pos_weight
	#print(pos_weight)
	optimizer = optim.Adam(model.parameters(),lr = 0.001)
	# print("Pos_Weight:")
	# print(pos_weight)
	criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
	total_loss = 0
	torch.autograd.set_detect_anomaly(True)

	train_accs = list()
	test_accs = list()

	for epoch in trange(epochs):
		#feed each sequence one at a time
		# for i in trange(train_X.size(0)):
		# 	model.zero_grad()
		# 	#print(train_X[i].unsqueeze(0).shape)    #inserted dim of 1 to 0th dimension
		# 	pred = model.forward(train_X[i].unsqueeze(0),1)
		# 	#print(pred)
		# 	#print(train_y[i].shape)
		# 	loss = criterion(pred,train_y[i])
		# 	loss.backward()
		# 	optimizer.step()
		# 	#print(loss.item())
		# 	total_loss += loss.item()
		# 	#predicted = torch.max(pred, 1)[1]
		#     #print(pred.detach())
		# 	#print(train_y[i])

		#feed entire batch all at once
		model.zero_grad()
		#print(train_X[i].unsqueeze(0).shape)    #inserted dim of 1 to 0th dimension
		pred = model.forward(train_X.float(),1)
		#print(pred)
		#print(train_y[i].shape)
		loss = criterion(pred,train_y.float())
		loss.backward()
		optimizer.step()
		#print("Epoch {} ".format(epoch) + "loss: {}".format(loss.item()))
		total_loss += loss.item()

		global test_X
		global test_y

		train_accs.append(model_accuracy(model,train_X,train_y,False))
		test_accs.append(model_accuracy(model,test_X,test_y,True))

	#print("End Of Training")
	return model, train_accs, test_accs


def model_accuracy(model, X, y,Test_flag):
	model.eval()
	pred = model.forward(X.float(),1)
	predicted = torch.max(pred,1)[1]
	actual = torch.max(y.float(),1)[1]
	correct = ((predicted == actual) == True).sum()
	total = X.size(0)

	# if Test_flag:
	# 	print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))
	# else:
	# 	print('Accuracy of the network on the train set: %d %%' % (100 * correct / total))

	model.train()
	return (correct/total)*100


if __name__ == '__main__':
	epochs = 200
	reshuffle = 3
	hidden_dim = 50

	accs = list() #list of final accuracies
	all_accs = list()
	for i in trange(reshuffle):  #reshuffle each loop
		#prepare data
		Combined_X, Combined_y = create_dataset('ExpertSamplesG2.csv','NoviceSamplesG2.csv')
		train_X,test_X,train_y,test_y = shuffle_and_split(Combined_X, Combined_y,0.7,i+52)
		# print("\nTrain Shape:")
		# print(train_X.size())

		#train and evaluate model
		model = Classifier(train_X.size(2),hidden_dim,1,2,0) #input dim, hidden dim, num_layers, output dim, dropout ratio
		model,train_accs,test_accs = train_model(model,train_X,train_y,epochs) # model, X, y, epochs
		acc = model_accuracy(model,test_X,test_y,True)
		accs.append(acc.item())
		all_accs.append([train_accs,test_accs])
		
		# #display number of params
		# model_parameters = filter(lambda p: p.requires_grad, model.parameters())
		# params = sum([np.prod(p.size()) for p in model_parameters])
		# print(params)
	print(accs)
	print("Mean: {}".format(st.mean(accs)))
	#print("stddev: {}".format(st.stdev(accs)))

	plt.xlabel("Epochs")
	plt.ylabel("Accuracy (%)")

	all_accs = np.array(all_accs)
	for i in range(len(all_accs)):
		plt.plot(np.transpose(all_accs[i,0]),label='Train', color = (0.5+np.random.random()*0.5,np.random.random()*0.1,np.random.random()*0.1))
		plt.plot(np.transpose(all_accs[i,1]),label='Test', color = (np.random.random()*0.1,0.5+np.random.random()*0.5,np.random.random()*0.1))
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


	#save trained model
	__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
	PATH = __location__ + '/LSTM.pth'
	torch.save(model.state_dict(), PATH)
	PATH = __location__ + '/fig.png'
	plt.savefig(PATH,dpi=250)