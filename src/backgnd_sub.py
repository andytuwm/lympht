import cv2


class BackgroundSubtractor:
    def __init__(self):
        self.background_set = False
        self.bg = None

    def set_frame_as_background(self, frame):
        self.bg = frame
        self.background_set = True

    def get_diff(self, frame):
        diff = cv2.absdiff(self.bg, frame)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)
        # blur > binary thresh
        blur = cv2.GaussianBlur(diff, (15, 15), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, None)
        # thresh = cv2.dilate(thresh, None, iterations=2)

        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(thresh, 254, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return thresh, contours
