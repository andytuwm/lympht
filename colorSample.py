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

            # Determine color sample locations
            sample_rows = 2
            sample_columns = 5
            color_sample_locations = []
            for i in range(sample_columns):
                for j in range(sample_rows):
                    x_location = frame_width / sample_columns * (i + 1) - frame_width / sample_columns / 2
                    y_location = (frame_height / 3) + (frame_height / 4) * j
                    color_sample_locations.append([x_location, y_location])
                    # Draw rectangles for sampling locations

                    # TODO: PUT THIS CODE LATER SO WE DONT SAMPLE THIS
                    # cv2.rectangle(frame, (x_location, y_location), (x_location + 10, y_location + 10), (0, 255, 0), 3)
                    # print(str(x_location) + " " + str(y_location))

            # print(color_sample_locations)
            color_samples = []
            # print(color_sample_locations)
            # print(frame_width, frame_height)

            for i in range(len(color_sample_locations)):
                x = color_sample_locations[i][0]
                y = color_sample_locations[i][1]
                # print(x)
                # print(x, y)
                # print(y,x)
                # x=200
                # print(type(x))
                # y=1000
                roi = frame[y:y + 10, x:x + 10]  # roi means region of interest

                # print(roi)

                if roi.shape[0] == 0:
                    print(roi)
                    # print(y,x)
                    # print(frame_width,frame_height)
                average_rows = np.average(roi, axis=0)
                average = np.average(average_rows, axis=0)
                color_samples.append(average)
                print average
            print "======"

            # print(color_samples)

            # filters everything within average ranges and adds it to sumFrames
            for color_sample in color_samples:
                # TODO check if this is alright, should be. i shortened it to a list comprehension,
                # then used a map to cap the values, and converted back to tuple. If it's good remove the commented stuff out below.
                upperbound = tuple(map(lambda n: min(n, 255), [x + 50 for x in color_sample]))
                lowerbound = tuple(map(lambda n: max(n, 0), [x - 50 for x in color_sample]))

            # for i in range(len(color_samples)):
            #     upperbound = (color_samples[i][0] + 50,
            #                   color_samples[i][1] + 50,
            #                   color_samples[i][2] + 50)
            #
            #     lowerbound = (color_samples[i][0] - 50,
            #                   color_samples[i][1] - 50,
            #                   color_samples[i][2] - 50)
            #
            #     # create a new tuple if upper and lower bounds are outside
            #     # of 0 to 255
            #     upperboundList = list(upperbound)
            #     lowerboundList = list(lowerbound)
            #
            #     for upbound, lowbound in zip(upperboundList, lowerboundList):
            #         if upbound > 255: upbound = 255
            #         if lowbound < 0: lowbound = 0
            #
            #     upperbound = tuple(upperboundList)
            #     lowerbound = tuple(lowerboundList)

                _ = cv2.inRange(frame, lowerbound, upperbound, frameRange)
                _ = cv2.inRange(frame, lowerbound, upperbound, frameRange2)
                _ = cv2.add(frameRange2, frameRange, sumFrames)

            # paint the averaging points after measurements have already been taken
            for vertex in color_sample_locations:
                x_location, y_location = vertex
                cv2.rectangle(frame, (x_location, y_location), (x_location + 10, y_location + 10), (0, 255, 0), 3)
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
