from clusterer import Clusterer, Plate
from imutils.video import WebcamVideoStream
from license_plate_ocr import ocr
from license_plate_detection import license_detection
from os.path import basename, splitext
from src.keras_utils import load_model
from src.utils import image_files_from_folder
from src.collections_utils import DiscardQueue
from vehicle_detection import vehicle_detect
from gen_outputs import generate_output
import argparse
import concurrent.futures
import cv2
import darknet.python.darknet as dn
import glob
import imutils
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

def produce_frame(video_stream, img_queue, end_event):
    """
    @summary: Read frame from provided stream and save it to provided queue
    """
    while not end_event.is_set():
        frame = video_stream.read()
        img_queue.put(frame)  


def display_frame(display_queue, end_event):
    while not end_event.is_set():
        frame = display_queue.get()
        frame = imutils.resize(frame, width=600)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def plate_legit(lp_str):
    if not lp_str:
        return False
    if len(lp_str) < 6:
        return False

    return True


def main():
    img_queue = DiscardQueue(QUEUE_SIZE)
    display_queue = queue.Queue()
    end_event = threading.Event()
    vs = WebcamVideoStream(src=0).start()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    executor.submit(produce_frame, vs, img_queue, end_event)
    executor.submit(display_frame, display_queue, end_event)

    overlay_remain_time = 1  # how many seconds plates remain in the image
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

    clusterer = Clusterer()
    timestamp_file = open("timestamp.txt","a+")
    try:
        while(True):
            labels = []
            platez = []

            img = img_queue.get()

            Icars, Lcars = vehicle_detect(
                img, vehicle_net, vehicle_meta, vehicle_threshold
            )

            for i, car_img in enumerate(Icars):
                # LPD
                lp_img, lp_label, ok = license_detection(car_img, wpod_net, lp_threshold)
                if not ok:
                    print("label not detected")
                    labels.append((Lcars[i], None, None))
                    continue
                # OCR
                lp_str = ocr(lp_img, ocr_net, ocr_meta, ocr_threshold)
                if plate_legit(lp_str):
                    labels.append((Lcars[i], lp_label, lp_str))
                    platez.append(Plate(lp_img, lp_str))
                else:
                    labels.append((Lcars[i], lp_label, None))

            # TODO: timestamp
            frame_ready = generate_output(img, labels, timestamp_file)
            clusterer.add_overlays(frame_ready, platez)
            display_queue.put(frame_ready)
    except KeyboardInterrupt:
        pass

    end_event.set() # finish
    executor.shutdown(wait=False)
    cv2.destroyAllWindows()
    vs.stop()
    timestamp_file.close()

    print("Execution time: %fs" % (time.time() - start_time))


if __name__ == "__main__":
    main()
