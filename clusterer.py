import numpy
from sklearn.cluster import AffinityPropagation
import distance

MAX_DISTANCE_INSIDE_CLUSTER = 4

class Clusterer:
	def __init__(self):
		self._all_platez = {} # {"plate txt": <plate cnt>, ...}
		self._clusters = {} # {"exemplar": ["plate 1", "plate 2", ...], ...}


	def get_clusters(self):
		"""
		returns clusters dictionary
		"""
		return self._clusters


	def add_platez(self, platez):
		"""
		adds platez to list of all known platez (if plate is not there)
		platez: new platez to add, array of strings
		"""
		for plate in platez:
			if plate not in self._all_platez:
				self._all_platez[plate] = 1
			else:
				self._all_platez[plate] += 1


	def make_clusters(self):
		"""
		stolen from: https://stats.stackexchange.com/questions/123060/clustering-a-long-list-of-strings-words-into-similarity-groups
		returns: array of clusters (arrays)
		"""
		if len(self._all_platez) < 1:
			return
		words = numpy.asarray(list(self._all_platez.keys())) # so that indexing with a list will work
		lev_similarity = -1 * numpy.array([[distance.levenshtein(w1, w2) for w1 in words] for w2 in words])

		affprop = AffinityPropagation(affinity="precomputed", preference=-MAX_DISTANCE_INSIDE_CLUSTER)
		affprop.fit(lev_similarity)
		clusters = {}
		for cluster_id in numpy.unique(affprop.labels_):
			exemplar = words[affprop.cluster_centers_indices_[cluster_id]] # one of the words
			cluster_members = numpy.unique(words[numpy.nonzero(affprop.labels_ == cluster_id)])
			clusters[exemplar] = cluster_members.tolist()
		self._clusters = clusters


	def dump_platez(self):
		"""prints all detected (unique) platez"""
		for plate in self._all_platez:
			print(plate)


	def dump_clusters(self, show_distances=False, show_exemplar=False):
		"""prints clusters of platez
		show_distances: will display distance matrix if True"""
		for exemplar, cluster in self._clusters.items():
			if show_exemplar:
				print(exemplar + ":")
			else:
				print("======================")
			for plate in cluster:
				if show_distances:
					distances = ""
					for plate2 in cluster:
						dist = distance.levenshtein(plate2, plate)
						distances += " %d" % dist
					print("\t%dx %s: %s" % (self._all_platez[plate], plate, distances))
				else:
					print("\t%dx %s" % (self._all_platez[plate], plate))
