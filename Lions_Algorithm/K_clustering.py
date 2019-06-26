import random
import numpy as np

class K_Cluster:

	def __init__(self, k):
		self.k = k

	def EuclideanDist(self, v1, v2):
		return np.sqrt(sum(pow(v1 - v2, 2)))

	def CalcCentroid(self, data):
		return np.mean(data, axis=0)

	def Initialize(self, data, initial_centroids):
		self.groups = np.full(len(data), -1).astype(int)
		self.centroids = initial_centroids
		if len(self.centroids) == 0:
			random_args = random.sample(range(len(data)), self.k)
			self.centroids = data[random_args]

	def Assign(self, data):
		for i in range(len(data)):
			distances = np.full(self.k, np.inf)
			for j in range(self.k):
				distances[j] = self.EuclideanDist(data[i], self.centroids[j])
			self.groups[i] = distances.argmin()
		for i in range(self.k):
			if i not in self.groups:
				self.groups[np.random.randint(0, len(data))] = i

	def Update(self, data):
		for i in range(self.k):
			self.centroids[i] = self.CalcCentroid(data[self.groups == i])

	def CalcGroups(self, data, initial_centroids):
		last_groups = np.full(len(data), -2).astype(int)
		self.Initialize(data, initial_centroids)
		while not (self.groups == last_groups).all():
			last_groups = np.copy(self.groups)
			self.Assign(data)
			self.Update(data)

	def GetGroups(self, data, initial_centroids=[]):
		grouped_data = []
		self.CalcGroups(data, initial_centroids.astype(np.float64))
		for i in range(self.k):
			grouped_data += [data[self.groups == i]]
		return grouped_data