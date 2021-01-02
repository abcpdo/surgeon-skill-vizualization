from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

dataset = loadtxt('pima-indians-diabetes.data.txt', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]

