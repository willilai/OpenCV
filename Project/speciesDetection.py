from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2

def plot_histogram(image, title, mask = None):
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    for (chan, color) in zip (chans, colors):
        hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
        plt.plot(hist, color = color)
        plt.xlim([0, 256])

    plt.show()
    cv2.waitKey(0)

def imageManipulation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    sobelX = cv2.Sobel(blurred, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(blurred, cv2.CV_64F, 0, 1)

    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))

    sobelCombined = cv2.bitwise_or(sobelX, sobelY)

    edged = cv2.Canny(sobelCombined, 10, 70)
    cv2.imshow("Edges", edged)

    return edged

def morphManip(edged):
     #morphology ops
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23,23))
    kernel2 = np.ones((15,15), np.uint8)
    #eliplical erode to get rid of some of the noise
    openFrame = cv2.dilate(edged, kernel2, iterations = 1)
    erodedFrame = cv2.erode(openFrame, kernel1, iterations = 1)
    cv2.imshow("morhp ops", erodedFrame)

    return erodedFrame

def contours(erodedFrame, image):
    (cnts, _) = cv2.findContours(erodedFrame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    whale = image.copy()
    c = max(cnts, key=cv2.contourArea)
    cv2.drawContours(whale, c, -1, (0, 255, 0), 2)
    cv2.imshow("Whale", whale)

    return c

def applyMask(c, image):
    mask = np.zeros(image.shape[:2], dtype = "uint8")
    cv2.fillConvexPoly(mask, c, (255, 255, 255))
    cv2.imshow("Mask", mask)
    cv2.imshow("Masked Whale", cv2.bitwise_and(image, image, mask = mask))

    return mask

def histofOImage():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to the Image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    cv2.imshow("Original", image)

    edged = imageManipulation(image)
    erodedFrame = morphManip(edged)
    c = contours(erodedFrame, image)
    mask = applyMask(c, image)

    plot_histogram(image, "Histogram for Masked Image", mask = mask)

def histofNImage():
    
def main():
    histofOImage()
    histofNImage()


main()
