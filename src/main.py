
import numpy as np

dataset1_filepath = "data/dataset1_normalised.txt"
dataset2_filepath = "data/dataset2_normalised.txt"

def get_dataset(filepath):
	"""Return randomised X, Y for the filepath supplied """

	with open(filepath, 'r') as fp:
		data = fp.readlines()

	data = data[1:]
	data = [i.strip().split('\t')[1:] for i in data]
	data = [map(float, i) for i in data]
	data = np.array(data)

	X = data[:,1:]
	Y = data[:,0]
	Y = np.asarray(Y, dtype=int)

	# randomise the data
	permutation_array = np.random.permutation(len(X))
	X = X[permutation_array]
	Y = Y[permutation_array]

	return X, Y




if __name__ == "__main__":
	pass
