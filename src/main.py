
import numpy as np
from scipy.cluster.vq import whiten , kmeans, vq
from sklearn.metrics import adjusted_rand_score
from sklearn import cluster

dataset1_filepath = "data/dataset1_normalised.txt"
dataset2_filepath = "data/dataset2_normalised.txt"

def randomise_data(X, Y):
	permutation_array = np.random.permutation(len(X))
	X = X[permutation_array]
	Y = Y[permutation_array]

	return X, Y

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
	X, Y = randomise_data(X, Y)

	return X, Y

def get_kmeans_clusters(X, Y):
	# no of clusters = number of unique values in Y
	k = len(np.unique(Y))

	# whitened = whiten(X)
	codebook, _ = kmeans(X, k)
	clusters, _ = vq(X, codebook)

	return clusters

def get_spectral_clusters(X, Y):
	# no of clusters = number of unique values in Y
	k = len(np.unique(Y))

	spectral = cluster.SpectralClustering(n_clusters=k, eigen_solver='arpack', affinity="nearest_neighbors")
	clusters = spectral.fit_predict(X)

	return clusters

def get_adjusted_rand_score(labels_true, label_pred):
	return adjusted_rand_score(labels_true, label_pred)

def get_score(X, Y):
	c = get_spectral_clusters(X, Y)
	return get_adjusted_rand_score(Y, c)


if __name__ == "__main__":
	for _ in range(10):
		X, Y = get_dataset(dataset1_filepath)
		pred = get_spectral_clusters(X, Y)
		# pred = get_kmeans_clusters(X, Y)
		print get_adjusted_rand_score(Y, pred)

