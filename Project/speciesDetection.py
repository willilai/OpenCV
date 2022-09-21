import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the Image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
blurred = cv2.GaussianBlur(image, (11, 11), 0)
cv2.imshow("Image", image)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)
