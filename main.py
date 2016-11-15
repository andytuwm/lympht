import cv2
import numpy as np
from src.utils import image_utils


class Lympht:

    isBackgroundSet = False
    background = None
    font = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        self.main_window_name = "lympht"

    def run(self):
        _, frame = self.capture.read()
        frame = image_utils.mirror_image(frame)

        while True:
            _, frame = self.capture.read()
            frame = image_utils.mirror_image(frame)

            # Listen for ESC or ENTER key
            c = cv2.waitKey(7) % 0x100
            if c == 27 or c == 10:
                break
            # On 'b' keypress, we save the background
            elif c == ord('b'):
                background = frame
                # cv2.imshow('background', background)
                self.isBackgroundSet = True

            # If background is set, we can differentiate
            # foreground and background
            if self.isBackgroundSet == True:
                diff = cv2.absdiff(background, frame)

                diff = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)
                # blur > binary thresh
                blur = cv2.GaussianBlur(diff, (15,15), 0)
                _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, None)
                # thresh = cv2.dilate(thresh, None, iterations=2)

                thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(thresh, 254, 255, cv2.THRESH_BINARY)

                _, contours, _ = cv2.findContours(thresh.copy(),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                contourCount = len(contours)
                print(contourCount)

                # find largest contour
                largestContourIndex = 0
                for i in range(1, len(contours)):
                    if(len(contours[i]) > len(contours[largestContourIndex])):
                        largestContourIndex = i

                if(contourCount > 0):
                    hull = cv2.convexHull(contours[largestContourIndex])
                    count, _, _ = hull.shape
                    hull.ravel()
                    hull.shape = count, 2
                    cv2.polylines(frame, np.int32([hull]), True, (0, 255, 0), 3)

                cv2.drawContours(frame, contours, largestContourIndex,
                    (255, 255, 0), 3)
                cv2.imshow('thresh', thresh)

            cv2.imshow(self.main_window_name, frame)

        cv2.destroyAllWindows()

    # creates a 'fading effect'
    def displayMovingAverage(self, frame, avg1, avg2):
        # do the moving average
        cv2.accumulateWeighted(frame, avg1, 1)
        cv2.accumulateWeighted(frame, avg2, 0.01)

        res1 = cv2.convertScaleAbs(avg1)
        res2 = cv2.convertScaleAbs(avg2)

        cv2.imshow('avg1', res1)
        cv2.imshow('avg2', res2)

if __name__ == "__main__":
    lympht = Lympht()
    lympht.run()
