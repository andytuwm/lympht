import cv2
import numpy as np


class ColorSampler:
    def __init__(self, frameHsv, sampleLocations):
        self.frameHsv = frameHsv
        self.colorSampleLocations = sampleLocations
        self.colorSampleAverages = []
        self.boundsList = []
        self.frameHeight, self.frameWidth = frameHsv.shape[:2]

    def addColorRangesFromFrame(self):
        # take the average of the colors within each RoI
        for sampleLocation in self.colorSampleLocations:
            x, y = sampleLocation
            roi = self.frameHsv[y:y + 10, x:x + 10]  # roi means region of interest
            averageRows = np.average(roi, axis=0)
            average = np.average(averageRows, axis=0)
            self.colorSampleAverages.append(average)

        # add a lower and upper bound for each averaged color sample
        for colorSample in self.colorSampleAverages:
            # TODO H has a range from 0 to 180 so adjust max value
            upperbound = tuple(map(lambda n: min(n, 255), [x + 10 for x in colorSample]))
            lowerbound = tuple(map(lambda n: max(n, 0), [x - 10 for x in colorSample]))
            self.boundsList.append([lowerbound, upperbound])

    # Output a mask that is a sum of the average colors sampled from the current frame
    def getColorMask(self, current_frame):
        rangeMask = np.zeros((self.frameHeight, self.frameWidth, 1), dtype=np.uint8)
        sumFrames = np.zeros((self.frameHeight, self.frameWidth, 1), dtype=np.uint8)

        # sum all the partial frames in the range of each color into sumFrames
        for lowBound, highBound in self.boundsList:
            _ = cv2.inRange(current_frame, lowBound, highBound, rangeMask)
            sumFrames = cv2.add(sumFrames, rangeMask)

        sumFrames = self.filterProcessing(sumFrames)
        return sumFrames

    # TODO Additional filtering
    def filterProcessing(self, frame):
        frame = cv2.GaussianBlur(frame, (15, 15), 0)
        kernel = np.ones((10, 10), np.uint8)
        #frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
        #frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        frame = cv2.erode(frame, kernel)
        frame = cv2.dilate(frame, kernel)
        return frame
