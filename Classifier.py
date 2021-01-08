from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
#from tensorflow.keras.utils import np_utils
#from tensorflow.keras.datasets import mnist
import numpy as np
import random
import os
from pandas import read_csv
from pandas import DataFrame


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
def load_file(filepath):
	dataframe = read_csv(filepath, header = None)
	return dataframe.to_numpy()  #returns as array

def load_samples(filepath):
	Output = []  #list of 2d arrays
	Samples = load_file(filepath)
	Two_D = np.empty((0,Samples.shape[1]))
	
	flag = True

	for i in range(Samples.shape[0]):
		if not np.isnan(Samples[i,0]):   #if the first element of each line is not NaN
			Two_D = np.vstack([Two_D,Samples[i,:]])
		else:
			Output.append(Two_D) #stack on the 2d array
			Two_D = np.empty((0,Samples.shape[1]))   #empty the 2d array
			
	return Output

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
Expert_Gestures = load_samples(os.path.join(__location__, 'ExpertSamples.csv'))
Novice_Gestures = load_samples(os.path.join(__location__, 'NoviceSamples.csv'))
Mixed_Gestures = random.shuffle(Expert_Gestures.append(Novice_Gestures),0.5)







