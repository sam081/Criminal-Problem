import numpy as np
import tensorflow as tf
import math
np.set_printoptions(suppress=True)

def load_dataset(filename,num_data_points):
	# load data from csv to numpy array
	data = np.genfromtxt(filename, delimiter= ',')
	
	#feature values of the training/test set -- per_id and output removed
	temp_data = data[1:num_data_points+1,:]
	train_data_X = data[1:num_data_points+1, 1:data.shape[1]-1]
	
	# Y for criminal or not
	train_data_Y = data[1:num_data_points+1, data.shape[1]-1]

	#person id
	person_id = data[1:,0]

    # divide data ie 45718 examples into train and dev/test set
	test_data_X = data[num_data_points+1:, 1:data.shape[1]-1]       # 1 to last -1 column    
	test_data_Y = data[num_data_points+1:, data.shape[1]-1]        # last column
    
	return temp_data,train_data_X, train_data_Y,test_data_X,test_data_Y,person_id


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0): 
	m = X.shape[1]                  # number of training examples
	mini_batches = []
	np.random.seed(seed)
	
	# Step 1: Shuffle (X, Y)
	permutation = list(np.random.permutation(m))
	shuffled_X = X[:, permutation]
	shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))
	# Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
	num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
	for k in range(0, num_complete_minibatches):
		mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
		mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
		
	# Handling the end case (last mini-batch < mini_batch_size)
	if m % mini_batch_size != 0:
		mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
		mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
		
	return mini_batches

def copy_data(data):
	x = data
	for i in range(0,len(data)):
		if data[i][71] == 1:
			z = data[i].reshape(1,72)
			x = np.concatenate((x,z),axis=0)
			
	return x
