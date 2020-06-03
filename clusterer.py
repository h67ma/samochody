import glob
import os
import cv2

OVERLAY_REMAIN_TIME = 0.5  # how many seconds plates remain in the image

class Overlay:
	def __init__(self, path, frames_left):
		self.path = path
		self.frames_left = frames_left

def overlay_img(base_img_path, plates):
	base_img = cv2.imread(base_img_path)
	base_rows, _, _ = base_img.shape
	offsety = 0
	offsetx = 0
	for plate in plates:
		overlay = cv2.imread(plate.path)
		rows, cols, _ = overlay.shape
		base_img[offsety:rows+offsety, offsetx:cols+offsetx] = overlay
		offsety += rows
		if offsety >= base_rows:
			offsety = 0
			offsetx += cols
	cv2.imwrite(base_img_path, base_img)

def add_overlays(out_dir, trim_dir, fps):
	overlay_remain_frames = int(OVERLAY_REMAIN_TIME * fps)
	current_overlays = []

	for out_file in os.listdir(out_dir):
		platez_paths = glob.glob("%s/%s_*car_lp.png" % (trim_dir, out_file[:-11]))
		for plate_path in platez_paths:
			current_overlays.append(Overlay(plate_path, overlay_remain_frames))

		if len(current_overlays) > 0:
			overlay_img("%s/%s" % (out_dir, out_file), current_overlays)
			for overlay in current_overlays:
				overlay.frames_left -= 1
				if overlay.frames_left <= 0:
					current_overlays.remove(overlay)
