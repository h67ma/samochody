import cv2
import os
import shutil
import sys
import time
import glob
import collections

class Overlay:
	def __init__(self, path, frames_left):
		self.path = path
		self.frames_left = frames_left

def frame_time(fps, frame_n):
    timestamp = frame_n / fps
    hours = int(timestamp / 3600)
    minutes = int(timestamp / 60) % 60
    seconds = int(timestamp) % 60
    ms = int(( timestamp - int(timestamp) ) * 1000)
    return (hours, minutes, seconds, ms)

def split_video(in_video, out_dir, fps):
    vidcap = cv2.VideoCapture(in_video)
    success, image = vidcap.read()
    count = 0
    time = None
    
    while success:
        time = frame_time(fps, count)
        cv2.imwrite("%s/%02d-%02d-%02d-%04d.jpg" % (out_dir, time[0], time[1], time[2], time[3]), image)
        success, image = vidcap.read()
        count += 1

def get_fps(in_video):
    vidcap = cv2.VideoCapture(in_video)
    return vidcap.get(cv2.CAP_PROP_FPS)

def combine_video(in_dir, fps, out_video):
    images = [f for f in os.listdir(in_dir)]
    img = cv2.imread("%s/%s" % (in_dir, images[0]))
    size = (img.shape[1], img.shape[0])
    out = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    images.sort()

    for image in images:
        img = cv2.imread("%s/%s" % (in_dir, image))
        out.write(img)

    out.release()

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

if len(sys.argv) < 2:
    print("Pls provide input video name (with extension)")
    quit()

overlay_remain_time = 1 # how many seconds plates remain in the image
start_time = time.time()
in_video = sys.argv[1]
out_video = "%s_tagged.mp4" % (in_video.split('.')[0])
timestamp_file = "%s_timestamps.csv" % (in_video.split('.')[0])
in_dir = "tmp_in"
trim_dir = "tmp_trim"
out_dir = "tmp_out"
lp_model="data/lp-detector/wpod-net_update1.h5"
fps = get_fps(in_video)
overlay_remain_frames = int(overlay_remain_time * fps)

if os.path.exists(in_dir):
    print("Removing old tmp files")
    shutil.rmtree(in_dir)
    shutil.rmtree(trim_dir)
    shutil.rmtree(out_dir)
os.mkdir(in_dir)
os.mkdir(trim_dir)
os.mkdir(out_dir)

print("Splitting video")
split_video(in_video, in_dir, fps)

print("Processing images")
os.system("python vehicle-detection.py %s %s" % (in_dir, trim_dir))
os.system("python license-plate-detection.py %s %s" % (trim_dir, lp_model))
os.system("python license-plate-ocr.py %s" % (trim_dir))

print("Generating timestamp file...")
os.system("python gen-outputs.py %s %s > %s" % (in_dir, trim_dir, timestamp_file))

# move actual output images to out_dir, leave trimmed in trim_dir
os.system("mv %s/*_output.png %s" % (trim_dir, out_dir)) # I'm too lazy to do that in python

current_overlays = []

# put plates into out images
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

print("Combining video")
combine_video(out_dir, fps, out_video)

print("Removing tmp files")
shutil.rmtree(in_dir)
shutil.rmtree(trim_dir)
shutil.rmtree(out_dir)

print("Execution time: %fs" % (time.time() - start_time))
