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
        sumFrames = np.zeros(frame.shape, np.uint8)
        frameRange = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        frameRange2 = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        while True:
            _, frame = self.capture.read()
            frame = image_utils.mirror_image(frame)

            # Listen for ESC or ENTER key
            c = cv2.waitKey(7) % 0x100
            if c == 27 or c == 10:
                break

            cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            #Determine color sample locations
            sample_rows = 2
            sample_columns = 5
            color_sample_locations = []
            for i in range(sample_columns):
                for j in range(sample_rows):
                    x_location = frame_width/sample_columns*(i+1) - frame_width/sample_columns/2
                    y_location = (frame_height/2) - 50 + 50*j
                    color_sample_locations.append([x_location,y_location])
                    #Draw rectangles for sampling locations
                    cv2.rectangle(frame,(x_location, y_location)
                        ,(x_location+10, y_location+10), (0, 255, 0), 3)

            color_samples = []

            print(len(color_sample_locations))
            print(color_sample_locations)

            for i in range(9):
                x = color_sample_locations[i][1]
                y = color_sample_locations[i][0]
                x_offset = x + 10
                y_offset = y + 10

                average = frame[x:x_offset,y:y_offset]
                average = np.average(average, axis=1)
                average = np.average(average, axis=0)
                color_samples.append(average)

            print(color_samples)

            # filters everything within average ranges and adds it to sumFrames
            for i in range(len(color_samples)):

                upperbound = (color_samples[i][0] + 50,
                    color_samples[i][1] + 50,
                    color_samples[i][2] + 50)

                lowerbound = (color_samples[i][0] - 50,
                    color_samples[i][1] - 50,
                    color_samples[i][2] - 50)

                # create a new tuple if upper and lower bounds are outside
                # of 0 to 255
                upperboundList = list(upperbound)
                lowerboundList = list(lowerbound)

                for i in range(3):
                    if(upperboundList[i] > 255):
                        upperboundList[i] = 255
                    if(lowerboundList[i] < 0):
                        lowerboundList[i] = 0

                upperbound = tuple(upperboundList)
                lowerbound = tuple(lowerboundList)

                _ = cv2.inRange(frame, lowerbound, upperbound, frameRange)
                _ = cv2.inRange(frame, lowerbound, upperbound, frameRange2)
                _ = cv2.add(frameRange2, frameRange, sumFrames)

            cv2.imshow('sumFrames', frameRange2)
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
