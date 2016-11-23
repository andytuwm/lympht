import cv2


class ColorSampleLocation:
    def __init__(self, frame):
        self.sampleRows = 10
        self.sampleColumns = 1
        self.frameHeight, self.frameWidth = frame.shape[:2]
        self.colorSampleLocations = []
        self.drawToggle = True

    # Determine color sample locations
    def getColorSampleLocations(self):
        # TODO set custom set of sampling locations
        for i in range(self.sampleColumns):
            for j in range(self.sampleRows):
                xLocation = self.frameWidth / self.sampleColumns * (i + 1) - self.frameWidth / self.sampleColumns / 6
                yLocation = (self.frameHeight / 3) + (self.frameHeight / 20) * j
                self.colorSampleLocations.append([xLocation, yLocation])
        return self.colorSampleLocations

    # Draw a box around the sampling locations
    def drawSampleLocations(self, frame):
        if self.drawToggle:
            for xLocation, yLocation in self.colorSampleLocations:
                cv2.rectangle(frame, (xLocation, yLocation), (xLocation + 10, yLocation + 10), (0, 255, 0), 3)

    def toggleDisplay(self):
        self.drawToggle = not self.drawToggle
