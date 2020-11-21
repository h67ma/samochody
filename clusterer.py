import numpy
from sklearn.cluster import AffinityPropagation
import distance
import cv2
from Queue import deque

MAX_DISTANCE_INSIDE_CLUSTER = 3
DISP_LAST_CNT = 10

class Plate:
	def __init__(self, first_img):
		self.detections_cnt = 1
		self.imgs = []
		if first_img != None:
			self.imgs.append(first_img)


class Clusterer:
	def __init__(self, save_images=False):
		"""
		save_images: if True, Clusterer will save plate images to disk, otherwise will ignore them.
		"""
		self._all_platez = {} # {"plate txt": Plate, ...}
		self._clusters = {} # {"exemplar": ["plate 1", "plate 2", ...], ...}
		self._last_platez = deque(maxlen=DISP_LAST_CNT)
		self._save_images = save_images


	def _find_best_cluster_plate_in_all_clusters(self, query_plate):
		"""
		Finds a cluster that query_plate belongs to, then searches for
		best plate inside (plate with largest amount of detections).
		If a cluster cannot be found, None will be returned.
		Please note that there can be two or more "best" platez inside single cluster.
		In such case one of them will be returned.
		"""
		for cluster_platez in self._clusters.values():
			if query_plate in cluster_platez:
				return self._find_best_cluster_plate(cluster_platez)
		return None # no cluster found


	def _find_best_cluster_plate(self, platez):
		"""
		Finds a plate in a cluster that has the with largest amount of detections.
		Please note that there can be two or more "best" platez inside a cluster.
		In such case one of them will be returned.
		platez: list of platez in a cluster.
		"""
		couts_dict = {}
		for plate in platez:
			couts_dict[plate] = self._all_platez[plate].detections_cnt
		return max(couts_dict, key=couts_dict.get)


	def dump_last_detections(self, print_best_cluster_sample=True):
		"""
		Prints last DISP_LAST_CNT platez that were added to Clusterer.
		print_best_cluster: if True, will also print a sample from corresponding cluster
			with the largest amount of detections (most probable "real" plate)
		"""
		if print_best_cluster_sample:
			print("recently detected platez (more recent at the end) (best cluster sample in brackets):")
		else:
			print("recently detected platez (more recent at the end):")
		for detected_plate in self._last_platez:
			if print_best_cluster_sample:
				print("%s (%s)" % (detected_plate, self._find_best_cluster_plate_in_all_clusters(detected_plate)))
			else:
				print(detected_plate)


	def add_platez(self, platez):
		"""
		Adds platez to list of all known platez (if plate is not there).
		platez: new platez to add, array of tuples (plate_text, plate_image)
		"""
		for plate in platez:
			plate_text = plate[0]
			plate_img = plate[1]
			self._last_platez.append(plate_text)
			if plate_text not in self._all_platez:
				self._all_platez[plate_text] = Plate(plate_img)
			else:
				self._all_platez[plate_text].detections_cnt += 1
				if self._save_images:
					self._all_platez[plate_text].imgs.append(plate_img)


	def make_clusters(self):
		"""
		Stolen from: https://stats.stackexchange.com/questions/123060/clustering-a-long-list-of-strings-words-into-similarity-groups
		Returns: array of clusters (arrays)
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


	def dump_platez(self, print_strs=False):
		"""
		Saves all plate images to files (if save_images was set to True). Optionally also prints all plate strings.
		print_strs: whether to print plate strings
		"""
		for plate_str, plate_data in self._all_platez.items():
			if print_strs:
				print(plate_str)

			if self._save_images:
				i = 1
				for img in plate_data.imgs:
					filename = "%s_%d.jpg" % (plate_str, i)
					cv2.imwrite(filename, img)
					i += 1


	def dump_clusters(self, show_distances=False, show_exemplar=False):
		"""
		Prints clusters of platez
		show_distances: will display distance matrix if True
		"""
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
					print("\t%dx %s: %s" % (self._all_platez[plate].detections_cnt, plate, distances))
				else:
					print("\t%dx %s" % (self._all_platez[plate].detections_cnt, plate))
