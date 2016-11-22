import cv2
import numpy as np


class ColorSampler:
    def __init__(self, frame_hsv, sample_locations):
        self.frame_hsv = frame_hsv
        self.color_sample_locations = sample_locations
        self.color_sample_averages = []
        self.bounds_list = []
        self.frame_height, self.frame_width = frame_hsv.shape[:2]

    def addColorRangesFromFrame(self):
        # take the average of the colors within each RoI
        for sample_location in self.color_sample_locations:
            x, y = sample_location
            roi = self.frame_hsv[y:y + 10, x:x + 10]  # roi means region of interest
            average_rows = np.average(roi, axis=0)
            average = np.average(average_rows, axis=0)
            self.color_sample_averages.append(average)

        # add a lower and upper bound for each averaged color sample
        for color_sample in self.color_sample_averages:
            # TODO H has a range from 0 to 180 so adjust max value
            upperbound = tuple(map(lambda n: min(n, 255), [x + 10 for x in color_sample]))
            lowerbound = tuple(map(lambda n: max(n, 0), [x - 10 for x in color_sample]))
            self.bounds_list.append([lowerbound, upperbound])

    # Output a mask that is a sum of the average colors sampled from the current frame
    def get_color_mask(self, current_frame):
        range_mask = np.zeros((self.frame_height, self.frame_width, 1), dtype=np.uint8)
        sumFrames = np.zeros((self.frame_height, self.frame_width, 1), dtype=np.uint8)

        # sum all the partial frames in the range of each color into sumFrames
        for low_bound, high_bound in self.bounds_list:
            _ = cv2.inRange(current_frame, low_bound, high_bound, range_mask)
            sumFrames = cv2.add(sumFrames, range_mask)

        sumFrames = self._filter_processing(sumFrames)
        return sumFrames

    # TODO Additional filtering
    def _filter_processing(self, frame):
        frame = cv2.GaussianBlur(frame, (15, 15), 0)
        kernel = np.ones((10, 10), np.uint8)
        #frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
        #frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        frame = cv2.erode(frame, kernel)
        frame = cv2.dilate(frame, kernel)
        return frame
