import numpy as np
import cv2
import imutils
from imutils.video import FPS

cap = cv2.VideoCapture(0)

fps = FPS().start()
while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    frame = imutils.resize(frame, width=400)
    if fps._end:
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.update()
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
