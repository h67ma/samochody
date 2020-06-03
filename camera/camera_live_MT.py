# import the necessary packages
from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse
import imutils
import cv2
import time
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100,
                help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
                help="Whether or not frames should be displayed")
ap.add_argument("-c", "--cap", type=int, default=300,
                help="Fps cap")
args = vars(ap.parse_args())

# SINGLE THREADED PROCESSING

# # grab a pointer to the video stream and initialize the FPS counter
# print("[INFO] sampling frames from webcam...")
# stream = cv2.VideoCapture(0)
# fps = FPS().start()
# # loop over some frames
# while fps._numFrames < args["num_frames"]:
# 	# grab the frame from the stream and resize it to have a maximum
# 	# width of 400 pixels
# 	(grabbed, frame) = stream.read()
# 	frame = imutils.resize(frame, width=400)
# 	# check to see if the frame should be displayed to our screen
# 	if args["display"] > 0:
# 		cv2.imshow("Frame", frame)
# 		key = cv2.waitKey(1) & 0xFF
# 	# update the FPS counter
# 	fps.update()
# # stop the timer and display FPS information
# fps.stop()
# print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# # do a bit of cleanup
# stream.release()
# cv2.destroyAllWindows()

vs = WebcamVideoStream(src=0).start()

fps = FPS().start()
# loop over some frames...this time using the threaded stream
while fps._numFrames < args["num_frames"]:

    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    # check to see if the frame should be displayed to our screen
    if args["display"] > 0 and fps._end and fps.fps() < 30:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        fps.update()
    # update the FPS counter

    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
