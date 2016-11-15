from __future__ import print_function
import cv2
import numpy as np
from src.utils import image_utils
from src import backgnd_sub


class Lympht:

    isBackgroundSet = False
    background = None
    font = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        self.main_window_name = "lympht"
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bg_sub = backgnd_sub.BackgroundSubtractor()

    def run(self):
        while True:
            _, frame = self.capture.read()
            frame = image_utils.mirror_image(frame)

            # Listen for ESC or ENTER key
            c = cv2.waitKey(7) % 0x100
            if c == 27 or c == 10:
                break
            # On 'b' keypress, we save the background
            elif c == ord('b'):
                self.bg_sub.set_frame_as_background(frame)

            # If background is set, we can differentiate
            # foreground and background
            if self.bg_sub.background_set is True:
                thresh, contours = self.bg_sub.get_diff(frame)

                contour_count = len(contours)
                print(contour_count)

                # find largest contour
                try:
                    largest_contour_index, largest_contour = max(enumerate(contours), key=lambda x: len(x[1]))
                except ValueError:
                    largest_contour_index, largest_contour = 0, [[]]

                if contour_count > 0:
                    hull = cv2.convexHull(largest_contour)
                    count, _, _ = hull.shape
                    hull.ravel()
                    hull.shape = count, 2
                    cv2.polylines(frame, np.int32([hull]), True, (0, 255, 0), 3)

                cv2.drawContours(frame, contours, largest_contour_index, (255, 255, 0), 3)
                cv2.imshow('thresh', thresh)

            cv2.imshow(self.main_window_name, frame)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    lympht = Lympht()
    lympht.run()
