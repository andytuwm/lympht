import cv2


# takes an image and returns the flipped image. creates a mirror effect.
def mirror_image(img):
    return cv2.flip(img, 1)
