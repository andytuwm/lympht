import cv2
import numpy as np

from src.utils import image_utils


class Lympht:
    background = None
    font = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        self.main_window_name = "lympht"

    def run(self):
        _, frame = self.capture.read()
        frame = image_utils.mirror_image(frame)
        frame_height, frame_width, frame_channels = frame.shape
        sumFrames = np.zeros((frame_height, frame_width, 1), dtype=np.uint8)
        currentFrame = np.zeros((frame_height, frame_width, 1), dtype=np.uint8)
        test = np.zeros((frame_height, frame_width, 1), dtype=np.uint8)

        while True:
            _, frame_RGB = self.capture.read()
            frame_RGB = image_utils.mirror_image(frame_RGB)

            # Listen for ESC or ENTER key
            c = cv2.waitKey(7) % 0x100
            if c == 27 or c == 10:
                break

            frame_hsv = cv2.cvtColor(frame_RGB, cv2.COLOR_BGR2HSV)
            frame = frame_hsv

            # Determine color sample locations
            sample_rows = 10
            sample_columns = 1
            color_sample_locations = []
            for i in range(sample_columns):
                for j in range(sample_rows):
                    x_location = frame_width / sample_columns * (i + 1) - frame_width / sample_columns / 6
                    y_location = (frame_height / 3) + (frame_height / 20) * j
                    color_sample_locations.append([x_location, y_location])

            color_samples = []

            for i in range(len(color_sample_locations)):
                x = color_sample_locations[i][0]
                y = color_sample_locations[i][1]
                roi = frame[y:y + 10, x:x + 10]  # roi means region of interest
                average_rows = np.average(roi, axis=0)
                average = np.average(average_rows, axis=0)
                color_samples.append(average)
                print average
            print "======"

            bounds_list = []

            # filters everything within average ranges and adds it to sumFrames
            for color_sample in color_samples:
                # TODO check if this is alright, should be. i shortened it to a list comprehension,
                # TODO H has a range from 0 to 180 so adjust max value
                # then used a map to cap the values, and converted back to tuple. If it's good remove the commented stuff out below.
                upperbound = tuple(map(lambda n: min(n, 255), [x + 10 for x in color_sample]))
                lowerbound = tuple(map(lambda n: max(n, 0), [x - 10 for x in color_sample]))
                bounds_list.append([lowerbound,upperbound])

            sumFrames = np.zeros((frame_height, frame_width, 1), dtype=np.uint8)
            for bound in bounds_list:
                _ = cv2.inRange(frame,bound[0],bound[1],currentFrame)
                sumFrames = cv2.add(sumFrames, currentFrame)

            # TODO Additional filtering
            sumFrames = cv2.GaussianBlur(sumFrames, (15, 15), 0)
            kernel = np.ones((5, 5), np.uint8)
            sumFrames = cv2.morphologyEx(sumFrames, cv2.MORPH_CLOSE, kernel)
            sumFrames = cv2.morphologyEx(sumFrames, cv2.MORPH_OPEN, kernel)



            # paint the averaging points after measurements have already been taken
            for vertex in color_sample_locations:
                x_location, y_location = vertex
                cv2.rectangle(frame, (x_location, y_location), (x_location + 10, y_location + 10), (0, 255, 0), 3)

            cv2.imshow('sumFrames', sumFrames)
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
