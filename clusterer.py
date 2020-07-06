import glob
import os
import cv2

OVERLAY_REMAIN_TIME = 0.5  # how many seconds plates remain in the image
FONT_GREEN = (0, 255, 0)

class Overlay:
    def __init__(self, img, frames_left):
        self.img = img
        self.frames_left = frames_left

class Plate:
    def __init__(self, img, text):
        self.img = img
        self.text = text

class Clusterer:
    def __init__(self, fps):
        self.overlay_remain_frames = int(OVERLAY_REMAIN_TIME * fps)
        self.current_overlays = {}  # {"plate text": Overlay(), ...}

    def _overlay_img(self, base_img):
        base_rows, _, _ = base_img.shape
        offsety = 0
        offsetx = 0
        for plate_text, plate in self.current_overlays.items():
            rows, cols, _ = plate.img.shape
            base_img[offsety:rows+offsety, offsetx:cols+offsetx] = plate.img
            cv2.putText(base_img, plate_text, (offsetx + cols, offsety + rows), cv2.FONT_HERSHEY_SIMPLEX, 1, FONT_GREEN, 2)
            offsety += rows
            if offsety >= base_rows:
                offsety = 0
                offsetx += cols

    """
    img: image to add overlay to
    platez: array of Plate
    """
    def add_overlays(self, img, platez):
        for plate in platez:
            self.current_overlays[plate.text] = Overlay(plate.img, self.overlay_remain_frames)

        if len(self.current_overlays) > 0:
            self._overlay_img(img)
            for plate_text, overlay in self.current_overlays.items():
                overlay.frames_left -= 1
                if overlay.frames_left <= 0:
                    del self.current_overlays[plate_text]
