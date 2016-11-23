from __future__ import print_function

import cv2
import numpy as np

import backgroundSub as bgs
import colorSampleLocation as csl
import colorSampler as cs
from angleDerivation import AngleDerivation as ad
from utils import image_utils

class Lympht:
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        self.mainWindowName = "lympht"
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bgSub = bgs.BackgroundSubtractor()
        self.cs = None
        self.csl = csl.ColorSampleLocation(self.capture.read()[1])
        self.cs_locations = self.csl.get_color_sample_locations()

    def run(self):
        while True:
            _, frame = self.capture.read()
            frame = image_utils.mirror_image(frame)
            drawFrame = frame.copy()
            bg = frame.copy()
            frameHeight, frameWidth, _ = frame.shape
            frameHsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Listen for ESC or ENTER key
            c = cv2.waitKey(7) % 0x100
            if c == 27 or c == 10:
                break
            # On 'b' keypress, we save the background
            elif c == ord('b'):
                self.bgSub.set_frame_as_background(bg)
            # On 'c' keypress, we sample the colors
            elif c == ord('c'):
                self.cs = cs.ColorSampler(frameHsv, self.cs_locations)
                self.cs.addColorRangesFromFrame()
            # On 'a' keypress, we add additional color samples to the current samples
            elif c == ord('a'):
                self.cs.addColorRangesFromFrame()

            # If background is set, we can differentiate
            # foreground and background
            if self.bgSub.background_set is True:
                bgThresh, contours = self.bgSub.get_diff(bg)
                contourCount = len(contours)
                cv2.imshow('thresh', bgThresh)

            # If skin color is sampled, we can isolate the sampled colors in the image
            if self.cs is not None:
                thresh = self.cs.get_color_mask(frameHsv)
                cv2.imshow('color_thresh', thresh)

            if self.bgSub.background_set and self.cs is not None:
                combined = cv2.add(bgThresh, thresh)

                _, contours, _ = cv2.findContours(combined.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contourCount = len(contours)

                # find largest contour
                try:
                    largest_contour_index, largest_contour = max(enumerate(contours), key=lambda x: len(x[1]))
                except ValueError:
                    largest_contour_index, largest_contour = 0, [[]]

                if contourCount > 0:
                    hull = cv2.convexHull(largest_contour)
                    # count, _, _ = hull.shape
                    hull.ravel()
                    # hull.shape = count, 2
                    cv2.polylines(frame, np.int32([hull]), True, (0, 255, 0), 3)

                    area = cv2.contourArea(largest_contour)
                    cv2.putText(drawFrame, "largest contour area " + str(area) + "px",
                                (0, frameHeight / 6), self.font, 0.5, (50, 50, 255), 2)

                    hullArea = cv2.contourArea(hull)
                    cv2.putText(drawFrame, "hull area " + str(hullArea) + "px",
                                (0, frameHeight / 7), self.font, 0.5, (50, 50, 255), 2)

                    [vx, vy, x, y] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
                    lefty = int((-x * vy / vx) + y)
                    righty = int(((frameWidth - x) * vy / vx) + y)
                    
                    verticalLine = [(frameWidth / 2, 0), (frameWidth / 2, frameHeight - 1)]
                    contourLine = [(frameWidth - 1, righty), (0, lefty)]

                    verticalVector = (0, 1)
                    contourVector = (vx, vy)
                    angle = ad.findAngle(verticalVector, contourVector)
                    
                    cv2.line(drawFrame, contourLine[0], contourLine[1], (0, 255, 0), 2)
                    cv2.line(drawFrame, verticalLine[0], verticalLine[1], (0, 255, 0), 2)
                    cv2.putText(drawFrame, "angle " + str(angle) + " degrees",
                                (0, frameHeight / 5), self.font, 0.5, (50, 50, 255), 2)

                cv2.drawContours(drawFrame, contours, largest_contour_index, (255, 255, 0), 3)

                cv2.imshow('combined', combined)

            self.csl.draw_sample_locations(drawFrame)
            cv2.imshow(self.mainWindowName, drawFrame)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    lympht = Lympht()
    lympht.run()
