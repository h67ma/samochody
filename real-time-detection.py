from imutils.video import WebcamVideoStream
from imutils.video import FPS
from src.collections_utils import DiscardQueue
import argparse
import imutils
import cv2
import time
import threading
import concurrent.futures
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
    cnt = 0
    while not end_event.is_set():
        frame = video_stream.read()
        img_queue.put(cnt)  # just for testing
        cnt += 1
        time.sleep(0.01)


def consume_frame(img_queue, display_queue, end_event):
    """
    @summary: Run plate detection on first image from provided queue
    """
    while not end_event.is_set():
        img = img_queue.get()
        # do sth with this image :)
        # print('Working on: {}'.format(img))
        time.sleep(0.5) # omit some cases - we cannot process as fast as we rcv
        display_queue.put(img)  # save to display


def display_frame(display_queue, end_event):
    # TODO add display logic
    while not end_event.is_set():
        frame = display_queue.get()
        print(frame)
        

img_queue = DiscardQueue(QUEUE_SIZE)
display_queue = queue.Queue()
end_event = threading.Event()
vs = WebcamVideoStream(src=0).start()

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    executor.submit(produce_frame, vs, img_queue, end_event)
    executor.submit(consume_frame, img_queue, display_queue, end_event)
    executor.submit(display_frame, display_queue, end_event)

time.sleep(10)
end_event.set() # finish

# fps = FPS().start()
# while (true):
#     frame = vs.read()

#     if  fps._end and fps.fps() < 30:
#         // Tutaj wrzucenie do kolejki - frame
#         cv2.waitKey(1) & 0xFF
#         fps.update()

#     fps.stop()

cv2.destroyAllWindows()
vs.stop()