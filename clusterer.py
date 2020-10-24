import numpy
from sklearn.cluster import AffinityPropagation
import distance
from src.drawing_utils import put_text

TEXT_LINE_HEIGHT = 50

class Clusterer:
	def __init__(self):
		self.all_platez = []
		self.clusters = []

	def overlay_clusters(self, img):
		"""
		draws all current clusters on img (in place)
		"""
		current_y = TEXT_LINE_HEIGHT
		for cluster in self.clusters:
			platez_in_cluster = "    ".join(cluster)
			put_text(img, platez_in_cluster, 0, current_y)
			current_y += TEXT_LINE_HEIGHT


	def add_platez(self, platez):
		"""
		adds platez to list of all known platez (if plate is not there)
		platez: new platez to add, array of strings
		"""
		for plate in platez:
			if plate not in self.all_platez:
				self.all_platez.append(plate)

	def make_clusters(self):
		"""
		stolen from: https://stats.stackexchange.com/questions/123060/clustering-a-long-list-of-strings-words-into-similarity-groups
		words: array of strings to be clustered
		returns: array of clusters (arrays)
		"""
		if len(self.all_platez) < 1:
			return
		words = numpy.asarray(self.all_platez) # so that indexing with a list will work
		lev_similarity = -1 * numpy.array([[distance.levenshtein(w1, w2) for w1 in words] for w2 in words])

		affprop = AffinityPropagation(affinity="precomputed", damping=0.5)
		affprop.fit(lev_similarity)
		clusters = []
		for cluster_id in numpy.unique(affprop.labels_):
			# exemplar = words[affprop.cluster_centers_indices_[cluster_id]] # one of the words
			cluster_members = numpy.unique(words[numpy.nonzero(affprop.labels_ == cluster_id)])
			clusters.append(cluster_members.tolist())
		self.clusters = clusters

	def debug_dump(self):
		with open("all_platez.txt", "w") as f:
			for plate in self.all_platez:
				f.write(plate + "\n")
