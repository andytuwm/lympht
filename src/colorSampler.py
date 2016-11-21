import cv2
import numpy as np

class ColorSampler:

    def __init__(self):
        self.color_frame_set = False
        self.color_frame = None
        self.color_sample_locations = []
        self.color_sample_averages = []
        self.bounds_list = []
        self.frame_width = None
        self.frame_height = None


    def setColorSampleLocations(self,frame):
        self.frame_height, self.frame_width, _ = frame.shape
        self.color_sample_locations = []

        sample_rows = 10
        sample_columns = 1
        for i in range(sample_columns):
            for j in range(sample_rows):
                x_location = self.frame_width / sample_columns * (i + 1) - self.frame_width / sample_columns / 6
                y_location = (self.frame_height / 3) + (self.frame_height / 20) * j
                self.color_sample_locations.append([x_location, y_location])


    def setColorFrame(self,frame):
        self.color_frame = frame
        self.color_frame_set = True
        self.color_sample_averages = []
        for sample_location in self.color_sample_locations:
            x, y = sample_location[0], sample_location[1]
            roi = self.color_frame[y:y + 10, x:x + 10]  # roi means region of interest
            average_rows = np.average(roi, axis=0)
            average = np.average(average_rows, axis=0)
            self.color_sample_averages.append(average)

        self.bounds_list = []

        # filters everything within average ranges and adds it to sumFrames
        for color_sample in self.color_sample_averages:
            # TODO check if this is alright, should be. i shortened it to a list comprehension,
            # TODO H has a range from 0 to 180 so adjust max value
            # then used a map to cap the values, and converted back to tuple. If it's good remove the commented stuff out below.
            upperbound = tuple(map(lambda n: min(n, 255), [x + 10 for x in color_sample]))
            lowerbound = tuple(map(lambda n: max(n, 0), [x - 10 for x in color_sample]))
            self.bounds_list.append([lowerbound, upperbound])



    def get_color_mask(self,frame):
        # Determine color sample locations
        currentFrame = np.zeros((self.frame_height, self.frame_width, 1), dtype=np.uint8)
        sumFrames = np.zeros((self.frame_height, self.frame_width, 1), dtype=np.uint8)
        for bound in self.bounds_list:
            _ = cv2.inRange(frame, bound[0], bound[1], currentFrame)
            sumFrames = cv2.add(sumFrames, currentFrame)

        # TODO Additional filtering
        sumFrames = cv2.GaussianBlur(sumFrames, (15, 15), 0)
        kernel = np.ones((5, 5), np.uint8)
        sumFrames = cv2.morphologyEx(sumFrames, cv2.MORPH_CLOSE, kernel)
        sumFrames = cv2.morphologyEx(sumFrames, cv2.MORPH_OPEN, kernel)

        return sumFrames

    def draw_sample_locations(self,frame):
        # paint the averaging points after measurements have already been taken
        for vertex in self.color_sample_locations:
            x_location, y_location = vertex
            cv2.rectangle(frame, (x_location, y_location), (x_location + 10, y_location + 10), (0, 255, 0), 3)
