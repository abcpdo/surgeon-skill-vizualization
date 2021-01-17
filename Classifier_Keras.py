from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import numpy as np
import os
from pandas import read_csv
from pandas import DataFrame
import torch
import torch.nn as nn
import torch.nn.functional as F
 
def load_file(filepath):
	dataframe = read_csv(filepath, header = None)
	return dataframe.to_numpy()  #returns as array

def load_samples(filepath):
	Output = []  #list of 2d arrays
	Samples = load_file(filepath)
	Two_D = np.empty((0,Samples.shape[1]))

	for i in range(Samples.shape[0]):
		if not np.isnan(Samples[i,0]):   #if the first element of each line is not NaN
			Two_D = np.vstack([Two_D,Samples[i,:]])
		else:
			Output.append(Two_D) #stack on the 2d array
			Two_D = np.empty((0,Samples.shape[1]))   #empty the 2d array
			
	return Output #list of arrays

def load_data():
	#load data from csv
	__location__ = os.path.realpath(
		os.path.join(os.getcwd(), os.path.dirname(__file__)))

	Expert_Gestures = load_samples(os.path.join(__location__, 'ExpertSamples.csv'))
	Novice_Gestures = load_samples(os.path.join(__location__, 'NoviceSamples.csv'))
	#generate labels  1 = Expert 0 = Novice
	Combined_y = [1]*len(Expert_Gestures) + [0]*len(Novice_Gestures)

	#get max step count
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

def shuffle_and_split(Combined_X,Combined_y):
	#shuffle
	np.random.seed(np.random.randint(0,1000))
	np.random.shuffle(Combined_X)
	np.random.shuffle(Combined_y)
	#split
	train_percent = 0.7
	test_percent = 1-train_percent
	train_X, test_X = np.split(Combined_X,[int(train_percent*Combined_X.shape[0])])
	train_y,test_y = np.split(Combined_y,[int(train_percent*Combined_y.shape[0])])
	return train_X,test_X,train_y,test_y

def evaluate_model(trainX,trainy,testX,testy,epoch,units=100):
	verbose, epochs, batch_size = 0, epoch, 64
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(LSTM(units, input_shape=(n_timesteps,n_features)))
	model.add(Dropout(0.5))
	model.add(Dense(units, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy

def summarize_results(scores):
	print(scores)
	m, s = np.mean(scores), np.std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


######################################

Combined_X,Combined_y = load_data()
# run experiment a few times
scores = list()
for i in range(5):
	epoch = 1+i*4
	for r in range(20):
		train_X,test_X,train_y,test_y = shuffle_and_split(Combined_X,Combined_y)
		score = evaluate_model(train_X,train_y,test_X,test_y,epoch,100)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append((score,epoch))
   
scores = np.array(scores)
summarize_results(scores[:,0].tolist())

plt.scatter(scores[:,1].tolist(), scores[:,0].tolist())
plt.show()

