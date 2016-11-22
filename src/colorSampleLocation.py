import cv2


class ColorSampleLocation:
    def __init__(self, frame):
        self.sample_rows = 8
        self.sample_columns = 1
        self.frame_h, self.frame_w = frame.shape[:2]
        self.color_sample_locations = []
        self.draw_toggle = True

    # Determine color sample locations
    def get_color_sample_locations(self):
        # TODO set custom set of sampling locations
        for i in range(self.sample_columns):
            for j in range(self.sample_rows):
                x_location = self.frame_w / self.sample_columns * (i + 1) - self.frame_w / self.sample_columns / 6
                y_location = (self.frame_h / 3) + (self.frame_h / 20) * j
                self.color_sample_locations.append([x_location, y_location])
        return self.color_sample_locations

    # Draw a box around the sampling locations
    def draw_sample_locations(self, frame):
        if self.draw_toggle:
            for x_location, y_location in self.color_sample_locations:
                cv2.rectangle(frame, (x_location, y_location), (x_location + 10, y_location + 10), (0, 255, 0), 3)

    def toggle_display(self):
        self.draw_toggle = not self.draw_toggle
