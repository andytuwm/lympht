from __future__ import print_function

import cv2
import numpy as np

from src import backgnd_sub, colorSampleLocation
from src.utils import image_utils
from src import colorSampler as cs


class Lympht:
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        self.main_window_name = "lympht"
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bg_sub = backgnd_sub.BackgroundSubtractor()
        self.cs = None
        self.csl = colorSampleLocation.ColorSampleLocation(self.capture.read()[1])
        self.cs_locations = self.csl.get_color_sample_locations()

    def run(self):
        while True:
            _, frame = self.capture.read()
            frame = image_utils.mirror_image(frame)
            frame_height, frame_width, frame_channels = frame.shape
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Listen for ESC or ENTER key
            c = cv2.waitKey(7) % 0x100
            if c == 27 or c == 10:
                break
            # On 'b' keypress, we save the background
            elif c == ord('b'):
                self.bg_sub.set_frame_as_background(frame)
            elif c == ord('c'):
                self.cs = cs.ColorSampler(frame_hsv, self.cs_locations)
                self.cs.addColorRangesFromFrame()

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

                    area = cv2.contourArea(largest_contour)
                    cv2.putText(frame, "largest contour area " + str(area) + "px",
                                (0, frame_height / 6), self.font, 0.5, (50, 50, 255), 2)

                    hull_area = cv2.contourArea(hull)
                    cv2.putText(frame, "hull area " + str(hull_area) + "px",
                                (0, frame_height / 7), self.font, 0.5, (50, 50, 255), 2)

                cv2.drawContours(frame, contours, largest_contour_index, (255, 255, 0), 3)
                cv2.imshow('thresh', thresh)

            # If skin color is sampled, we can isolate the sampled colors in the image
            if self.cs is not None:
                # print(self.cs.color_sample_averages)
                color_thresh = self.cs.get_color_mask(frame_hsv)
                # self.cs.draw_sample_locations(frame)
                cv2.imshow('color_thresh', color_thresh)

            draw_frame = frame
            self.csl.draw_sample_locations(draw_frame)
            cv2.imshow(self.main_window_name, draw_frame)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    lympht = Lympht()
    lympht.run()
