import glob
import os
import cv2

OVERLAY_REMAIN_TIME = 0.5  # how many seconds plates remain in the image
FONT_GREEN = (0, 255, 0)

class Overlay:
	def __init__(self, path, frames_left):
		self.path = path
		self.frames_left = frames_left

def overlay_img(base_img_path, plates):
	base_img = cv2.imread(base_img_path)
	base_rows, _, _ = base_img.shape
	offsety = 0
	offsetx = 0
	for plate_text, plate in plates.items():
		overlay = cv2.imread(plate.path)
		rows, cols, _ = overlay.shape
		base_img[offsety:rows+offsety, offsetx:cols+offsetx] = overlay
		cv2.putText(base_img, plate_text, (offsetx + cols, offsety + rows), cv2.FONT_HERSHEY_SIMPLEX, 1, FONT_GREEN, 2)
		offsety += rows
		if offsety >= base_rows:
			offsety = 0
			offsetx += cols
	cv2.imwrite(base_img_path, base_img)

def add_overlays(out_dir, trim_dir, fps):
	overlay_remain_frames = int(OVERLAY_REMAIN_TIME * fps)
	current_overlays = {}  # {"plate text": Overlay(), ...}

	for out_file in os.listdir(out_dir):
		platez_str_paths = glob.glob("%s/%s_*car_lp_str.txt" % (trim_dir, out_file[:-11]))
		for plate_str_path in platez_str_paths:
			with open(plate_str_path, 'r') as f:
				text = f.readline()
				current_overlays[text] = Overlay("%s.png" % plate_str_path[:-8], overlay_remain_frames)

		if len(current_overlays) > 0:
			overlay_img("%s/%s" % (out_dir, out_file), current_overlays)
			for plate_text, overlay in current_overlays.items():
				overlay.frames_left -= 1
				if overlay.frames_left <= 0:
					del current_overlays[plate_text]
