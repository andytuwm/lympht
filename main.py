import math

import cv2
import numpy as np

from src.utils import image_utils

class Lympht:
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        self.main_window_name = "lympht"
        cv2.namedWindow(self.main_window_name, 1)
        # cv2.namedWindow("Threshold1", 1)
        # cv2.namedWindow("Threshold2", 1)
        # cv2.namedWindow("hsv", 1)

    def run(self):
        # initiate font
        font = cv2.FONT_HERSHEY_SIMPLEX

        frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        # instantiate images
        # hsv_img=cv2.CreateImage(cv2.GetSize(cv2.QueryFrame(self.capture)),8,3)
        hsv_img = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        threshold_img1 = np.zeros((frame_height, frame_width, 1), dtype=np.uint8)
        # threshold_img1a = np.zeros((frame_height, frame_width, 1), dtype=np.uint8)
        threshold_img2 = np.zeros((frame_height, frame_width, 1), dtype=np.uint8)
        # threshold_img2a = np.zeros((frame_height, frame_width, 1), dtype=np.uint8)

        # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        # writer = cv2.VideoWriter("angle_tracking.avi", fourcc, 30, (frame_height, frame_width), 1)

        while True:
            # capture the image from the cam
            ret, img = self.capture.read()

            # get contour
            # reti, thresh = cv2.threshold(img,127,255,0)
            # contours, hierarchy = cv2.findContours(thresh,1,2)
            # cnt = contours[0]
            YELLOW_MIN = np.array([20, 120, 120], np.uint8)
            YELLOW_MAX = np.array([30, 200, 200], np.uint8)

            BLUE_CONTOUR_MIN = np.array([110, 50, 50], np.uint8)
            BLUE_CONTOUR_MAX = np.array([130, 255, 255], np.uint8)

            BLUE_MIN = np.array([0, 0, 0], np.uint8)
            BLUE_MAX = np.array([255, 255, 255], np.uint8)

            # convert the image to HSV
            cv2.blur(img, (frame_height, frame_width))
            cv2.medianBlur(img, 5)
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # define range of color in HSV

            # threshold the image to isolate two colors
            mask_blue = cv2.inRange(hsv_img, BLUE_MIN, BLUE_MAX, threshold_img1)
            mask_yellow = cv2.inRange(hsv_img, YELLOW_MIN, YELLOW_MAX, threshold_img2)

            # Markers
            cv2.circle(img, (500, 600), 2, (0, 255, 0), 20)
            cv2.circle(img, (697, 600), 2, (0, 255, 0), 20)

            im2, contoursYellow, hierarchy2 = cv2.findContours(threshold_img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            im1, contoursBlue, hierarchy1 = cv2.findContours(threshold_img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            #            filtedBlueContours = []
            #            for c in contoursBlue:
            #                if(len(c) > 100):
            #                    print "contour: "
            #                    print len(c)
            #                    filtedBlueContours.append(c)

            cv2.drawContours(img, contoursYellow, -1, (255, 255, 0), 3)
            cv2.drawContours(img, contoursBlue, -1, (255, 0, 0), 3)

            # determine the moments of the two objects
            moments1 = cv2.moments(threshold_img1)
            moments2 = cv2.moments(threshold_img2)
            area1 = moments1['m00']
            area2 = moments2['m00']
            # countourArea1 = cv2.contourArea(contoursBlue)
            # countourArea2 = cv2.contourArea(cnt)
            # print area1 #Blue
            # print area2
            # print "Contour Area: " + contourArea1
            # print contourArea2

            # initialize x and y
            x1, y1, x2, y2 = (1, 2, 3, 4)
            coord_list = [x1, y1, x2, y2]
            for x in coord_list:
                x = 0

            # there can be noise in the video so ignore objects with small areas
            if (area1 > 99999):  # 99999
                # x and y coordinates of the center of the object is found by dividing the 1,0 and 0,1 moments by the area
                x1 = int(moments1['m10'] / area1)
                y1 = int(moments1['m01'] / area1)

                # draw circle
                cv2.circle(img, (x1, y1), 2, (0, 255, 0), 20)

                # write x and y position
                cv2.putText(img, str(x1) + "," + str(y1), (x1, y1 + 20), font, 1, (255, 255, 255))  # Draw the text

            if (area2 > 9999):
                # x and y coordinates of the center of the object is found by dividing the 1,0 and 0,1 moments by the area
                x2 = int(moments2['m10'] / area2)
                y2 = int(moments2['m01'] / area2)

                # draw circle
                cv2.circle(img, (x2, y2), 2, (0, 255, 0), 20)

                cv2.putText(img, str(x2) + "," + str(y2), (x2, y2 + 20), font, 1, (255, 255, 255))  # Draw the text
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
            # draw line and angle
            # cv2.line(img,(x1,y1),(frame_height, frame_width, y1),(100,100,100,100),4,cv2.LINE_AA)
            x1 = float(x1)
            y1 = float(y1)
            x2 = float(x2)
            y2 = float(y2)
            if (x2 - x1 != 0):
                angle = int(math.atan((y1 - y2) / (x2 - x1)) * 180 / math.pi)
            else:
                print "zero detected"
                angle = None
            if (angle != None):
                cv2.putText(img, str(angle), (int(x1) + 50, (int(y2) + int(y1)) / 2), font, 8, (255, 255, 255), 8)

            # cv2.writeFrame(writer,img)

            # display frames to users
            cv2.imshow(self.main_window_name, image_utils.mirror_image(img))
            # cv2.imshow("Threshold1",threshold_img1)
            # cv2.imshow("Threshold2",threshold_img2) #Yellow
            # cv2.imshow("hsv",hsv_img)
            # Listen for ESC or ENTER key
            c = cv2.waitKey(7) % 0x100
            if c == 27 or c == 10:
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    t = Lympht()
    t.run()
