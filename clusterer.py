import numpy
from sklearn.cluster import AffinityPropagation
import distance

MAX_DISTANCE_INSIDE_CLUSTER = 4

class Clusterer:
	def __init__(self):
		self.all_platez = {}
		self.clusters = {}

	def add_platez(self, platez):
		"""
		adds platez to list of all known platez (if plate is not there)
		platez: new platez to add, array of strings
		"""
		for plate in platez:
			if plate not in self.all_platez:
				self.all_platez[plate] = 1
			else:
				self.all_platez[plate] += 1

	def make_clusters(self):
		"""
		stolen from: https://stats.stackexchange.com/questions/123060/clustering-a-long-list-of-strings-words-into-similarity-groups
		words: array of strings to be clustered
		returns: array of clusters (arrays)
		"""
		if len(self.all_platez) < 1:
			return
		words = numpy.asarray(list(self.all_platez.keys())) # so that indexing with a list will work
		lev_similarity = -1 * numpy.array([[distance.levenshtein(w1, w2) for w1 in words] for w2 in words])

		affprop = AffinityPropagation(affinity="precomputed", random_state=None, preference=-MAX_DISTANCE_INSIDE_CLUSTER)
		print(affprop.preference)
		affprop.fit(lev_similarity)
		clusters = {}
		for cluster_id in numpy.unique(affprop.labels_):
			exemplar = words[affprop.cluster_centers_indices_[cluster_id]] # one of the words
			cluster_members = numpy.unique(words[numpy.nonzero(affprop.labels_ == cluster_id)])
			clusters[exemplar] = cluster_members.tolist()
		self.clusters = clusters

	def dump_platez(self):
		for plate in self.all_platez:
			print(plate)

	def dump_clusters(self, show_distances):
		for exemplar, cluster in self.clusters.items():
			print(exemplar + ":")
			for plate in cluster:
				if show_distances:
					distances = ""
					for plate2 in cluster:
						dist = distance.levenshtein(plate2, plate)
						distances += " %d" % dist
					print("\t%dx %s: %s" % (self.all_platez[plate], plate, distances))
				else:
					print("\t%dx%s" % (self.all_platez[plate], plate))
