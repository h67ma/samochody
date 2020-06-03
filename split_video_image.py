import cv2
import os
import shutil
import sys
import time
import collections
from clusterer import add_overlays

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

if len(sys.argv) < 2:
    print("Pls provide input video name (with extension)")
    quit()

start_time = time.time()
in_video = sys.argv[1]
out_video = "%s_tagged.mp4" % (in_video.split('.')[0])
timestamp_file = "%s_timestamps.csv" % (in_video.split('.')[0])
in_dir = "tmp_in"
trim_dir = "tmp_trim"
out_dir = "tmp_out"
lp_model="data/lp-detector/wpod-net_update1.h5"
fps = get_fps(in_video)

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

add_overlays(out_dir, trim_dir, fps)

print("Combining video")
combine_video(out_dir, fps, out_video)

print("Removing tmp files")
shutil.rmtree(in_dir)
shutil.rmtree(trim_dir)
shutil.rmtree(out_dir)

print("Execution time: %fs" % (time.time() - start_time))
