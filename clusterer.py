import numpy
from sklearn.cluster import AffinityPropagation
import distance
import cv2
import os
import shutil
from Queue import deque
from src.drawing_utils import put_text

MAX_DISTANCE_INSIDE_CLUSTER = 3
DISP_LAST_CNT = 10
MIN_OCC_LAST = 2
LINE_HEIGHT = 35

class Plate:
	def __init__(self, first_img):
		self.detections_cnt = 1
		self.imgs = []
		if first_img is not None:
			self.imgs.append(first_img)


class Clusterer:
	def __init__(self, save_images=False):
		"""
		save_images: if True, Clusterer will save plate images to disk, otherwise will ignore them.
		"""
		self._all_platez = {} # {"plate txt": Plate, ...}
		self._clusters = {} # {"exemplar": ["plate 1", "plate 2", ...], ...}
		self._last_platez = deque(maxlen=DISP_LAST_CNT)
		self._last_platez_to_display = []
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


	def overlay_last_detections(self, img):
		yoffset = LINE_HEIGHT
		for line in self._last_platez_to_display:
			put_text(img, line, 0, yoffset)
			yoffset += LINE_HEIGHT


	def make_last_detections(self, print_best_cluster_sample=True):
		"""
		Prepares last DISP_LAST_CNT platez that were added to Clusterer to display later.
		print_best_cluster: if True, will also print a sample from corresponding cluster
			with the largest amount of detections (most probable "real" plate)
		"""
		self._last_platez_to_display.clear()
		if print_best_cluster_sample:
			self._last_platez_to_display.append("recently detected platez (more recent at the end) (best cluster sample in brackets):")
		else:
			self._last_platez_to_display.append("recently detected platez (more recent at the end):")
		for detected_plate in self._last_platez:
			if print_best_cluster_sample:
				self._last_platez_to_display.append("%s (%s)" % (detected_plate, self._find_best_cluster_plate_in_all_clusters(detected_plate)))
			else:
				self._last_platez_to_display.append(detected_plate)


	def add_platez(self, platez):
		"""
		Adds platez to list of all known platez (if plate is not there).
		platez: new platez to add, array of tuples (plate_text, plate_image)
		"""
		for plate in platez:
			plate_text = plate[0]
			plate_img = plate[1]

			if plate_text not in self._all_platez:
				self._all_platez[plate_text] = Plate(plate_img)
			else:
				self._all_platez[plate_text].detections_cnt += 1

				if self._save_images:
					self._all_platez[plate_text].imgs.append(plate_img)

				if self._all_platez[plate_text].detections_cnt >= MIN_OCC_LAST:
					self._last_platez.append(plate_text)


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


	def dump_platez_imgs(self):
		"""
		Saves all plate images to files (if save_images was set to True). Optionally also prints all plate strings.
		print_strs: whether to print plate strings
		"""
		if not self._save_images:
			return
		dir_name = "clusterlogs"
		if os.path.isdir(dir_name):
			shutil.rmtree(dir_name)
		os.makedirs(dir_name)
		for plate_str, plate_data in self._all_platez.items():
			i = 1
			for img in plate_data.imgs:
				filename = os.path.join(dir_name, "%s_%d.jpg" % (plate_str, i))
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
