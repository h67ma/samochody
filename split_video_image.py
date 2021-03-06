from clusterer import Clusterer
from debug_platez_overlay import DebugPlatezOverlay, Plate
from imutils.video import WebcamVideoStream
from license_plate_ocr import ocr
from license_plate_detection import license_detection
from os.path import basename, splitext
from src.drawing_utils import put_text
from src.keras_utils import load_model
from src.utils import image_files_from_folder
from src.collections_utils import DiscardQueue, DiscardList
from vehicle_detection import vehicle_detect
from gen_outputs import generate_output
import argparse
import concurrent.futures
import cv2
import darknet.python.darknet as dn
import glob
import imutils
import math
import os
import shutil
import sys
import threading
import time

try:
    import queue
except ImportError:
    import Queue as queue

QUEUE_SIZE = 10
FRAME_PER_FPS = 5
CLUSTER_EVERY_X_FRAMES = 30

def produce_frame(video_stream, img_queue, end_event):
    """
    @summary: Read frame from provided stream and save it to provided queue
    """
    while not end_event.is_set():
        frame = video_stream.read()
        img_queue.put(frame)  


def display_frame(display_queue, end_event):
    fps_queue = DiscardList(FRAME_PER_FPS)
    while not end_event.is_set():
        frame = display_queue.get()
        fps_queue.append(time.time())
        height, width, _ = frame.shape
        put_text(frame, str(int(math.ceil(1/fps_queue.get_avg()))), width-80, 50, 2)   # top right
        frame = imutils.resize(frame, width=800)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def plate_legit(lp_str):
    if not lp_str:
        return False
    if len(lp_str) < 6:
        return False

    return True

def time_now_ms():
    return int(round(time.time() * 1000))

def main():

    from_file = False
    if len(sys.argv) > 1:
        from_file = True
        in_filename = sys.argv[1]

        vs = cv2.VideoCapture(in_filename)

        fps = vs.get(5)
        width = vs.get(3)
        height = vs.get(4)
        print("fps=%f\nw=%d\nh=%d" % (fps, width, height))
        
        video_writer = cv2.VideoWriter(in_filename + "_out.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (int(width), int(height)))
    else:
        vs = WebcamVideoStream(src=0).start()
        img_queue = DiscardQueue(QUEUE_SIZE)
        display_queue = queue.Queue()
        end_event = threading.Event()

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        executor.submit(produce_frame, vs, img_queue, end_event)
        executor.submit(display_frame, display_queue, end_event)

    start_time = time.time()
    #timestamp_file = "%s_timestamps.csv" % (in_video.split(".")[0])
    lp_model = "data/lp-detector/wpod-net_update1.h5"

    print("Processing images")
    vehicle_threshold = 0.5

    vehicle_weights = "data/vehicle-detector/yolo-voc.weights"
    vehicle_netcfg = "data/vehicle-detector/yolo-voc.cfg"
    vehicle_dataset = "data/vehicle-detector/voc.data"

    print("Loading vehicle model...")
    vehicle_net = dn.load_net(vehicle_netcfg, vehicle_weights, 0)
    vehicle_meta = dn.load_meta(vehicle_dataset)

    ocr_threshold = 0.4
    ocr_weights = "data/ocr/ocr-net.weights"
    ocr_netcfg = "data/ocr/ocr-net.cfg"
    ocr_dataset = "data/ocr/ocr-net.data"

    print("Loading OCR model...")
    ocr_net = dn.load_net(ocr_netcfg, ocr_weights, 0)
    ocr_meta = dn.load_meta(ocr_dataset)

    lp_threshold = 0.5
    wpod_net_path = lp_model
    print("Loading wpod model...")
    wpod_net = load_model(wpod_net_path)

    #debug_overlay = DebugPlatezOverlay()
    clusterer = Clusterer(save_images=True)
    timestamp_file = open("timestamp.csv","a+")
    dont = 0
    try:
        while(True):
            if from_file:
                _, img = vs.read()
                if img is None:
                    break
            else:
                img = img_queue.get()

            labels = []
            #platez = []
            platez_for_clusterer = []

            Icars, Lcars = vehicle_detect(
                img, vehicle_net, vehicle_meta, vehicle_threshold
            )

            # LPD
            if Icars:
                license_plates = license_detection(Icars, wpod_net, lp_threshold)
                # OCR
                for i in range(0, len(license_plates)):
                    lp = license_plates[i]
                    if not lp[2]:
                        print("label not detected")
                        labels.append((Lcars[i], None, None))
                        continue
                    lp_str = ocr(lp[0], ocr_net, ocr_meta, ocr_threshold)
                    if plate_legit(lp_str):
                        labels.append((Lcars[i], lp[1], lp_str))
                        #platez.append(Plate(lp[0], lp_str))
                        platez_for_clusterer.append((lp_str, lp[0]))
                    else:
                        labels.append((Lcars[i], lp[1], None))


            # TODO: timestamp
            frame_ready = generate_output(img, labels, timestamp_file)
            #debug_overlay.add_overlays(frame_ready, platez)
            clusterer.add_platez(platez_for_clusterer)
            dont += 1
            if dont >= CLUSTER_EVERY_X_FRAMES:
                clusterer.make_clusters()
                clusterer.make_last_detections()
                clusterer.dump_clusters()
                clusterer.dump_platez_imgs()
                dont = 0
            clusterer.overlay_last_detections(img)
            if from_file:
                video_writer.write(frame_ready)
            else:
                display_queue.put(frame_ready)
    except KeyboardInterrupt:
        pass

    if not from_file:
        end_event.set() # finish
        executor.shutdown(wait=False)
        cv2.destroyAllWindows()
        vs.stop()
    timestamp_file.close()

    print("Execution time: %fs" % (time.time() - start_time))


if __name__ == "__main__":
    main()
