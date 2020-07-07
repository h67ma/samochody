import glob
import os
import cv2

FONT_GREEN = (0, 255, 0)
FRAMES_DISPLAYED = 30

# can be read from img but what's the point
PLATE_W = 240
PLATE_H = 80
MID_PLATE_TEXT = 50
INITIAL_Y_OFFSET = 40

class Overlay:
    def __init__(self, img, frames_left):
        self.img = img
        self.frames_left = frames_left
        self.occured = 1  # in series

class Plate:
    def __init__(self, img, text):
        self.img = img
        self.text = text

"""
wrapper for cv2 put text thing
"""
def put_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, FONT_GREEN, 2)

class Clusterer:
    def __init__(self):
        self.current_overlays = {}  # {"plate text": Overlay(), ...}
        self.total_detections = {}  # {"plate text": 123, ...}

    """
    adds all current overlays to img
    """
    def _overlay_img(self, img):
        put_text(img, "img", 0, 30)
        put_text(img, "text", 240, 30)
        put_text(img, "curr", 400, 30)
        put_text(img, "total", 480, 30)

        base_h, base_w, _ = img.shape

        offsety = INITIAL_Y_OFFSET
        offsetx = 0
        for plate_text, plate in self.current_overlays.items():
            # rows, cols, _ = plate.img.shape
            img[offsety:PLATE_H+offsety, offsetx:PLATE_W+offsetx] = plate.img
            put_text(img, plate_text, offsetx + PLATE_W, offsety + MID_PLATE_TEXT)
            put_text(img, str(plate.occured), offsetx + PLATE_W + 160, offsety + MID_PLATE_TEXT)
            put_text(img, str(self.total_detections[plate_text]), offsetx + PLATE_W + 240, offsety + MID_PLATE_TEXT)
            offsety += PLATE_H
            if offsety + PLATE_H >= base_h:
                offsety = INITIAL_Y_OFFSET
                offsetx += PLATE_W + 300
                if offsetx + PLATE_W >= base_w:
                    print("Can't fit all those platez in the image")
                    return

    """
    adds overlay to img with detected plates. plates detected earlier are also shown (for a while)
    img: current image
    platez: list of Plate() (plates and detected text)
    """
    def add_overlays(self, img, platez):
        for plate in platez:
            if plate.text in self.current_overlays:
                self.current_overlays[plate.text].img = plate.img
                self.current_overlays[plate.text].occured += 1
            else:
                self.current_overlays[plate.text] = Overlay(plate.img, FRAMES_DISPLAYED)

            if plate.text in self.total_detections:
                self.total_detections[plate.text] += 1
            else:
                self.total_detections[plate.text] = 1

        if len(self.current_overlays) > 0:
            self._overlay_img(img)
            for plate_text, overlay in self.current_overlays.items():
                overlay.frames_left -= 1
                if overlay.frames_left <= 0:
                    del self.current_overlays[plate_text]
