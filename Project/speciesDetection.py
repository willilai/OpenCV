from matplotlib import pyplot as plt
from statistics import mean
import numpy as np
import argparse
import cv2

def imageManipulation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    sobelX = cv2.Sobel(blurred, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(blurred, cv2.CV_64F, 0, 1)

    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))

    sobelCombined = cv2.bitwise_or(sobelX, sobelY)

    edged = cv2.Canny(sobelCombined, 10, 70)
    # cv2.imshow("Edges", edged)

    return edged

def morphManip(edged):
     #morphology ops
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23,23))
    kernel2 = np.ones((15,15), np.uint8)
    #eliplical erode to get rid of some of the noise
    openFrame = cv2.dilate(edged, kernel2, iterations = 1)
    erodedFrame = cv2.erode(openFrame, kernel1, iterations = 1)
    # cv2.imshow("morhp ops", erodedFrame)

    return erodedFrame

def contours(erodedFrame, image):
    (cnts, _) = cv2.findContours(erodedFrame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    whale = image.copy()
    c = max(cnts, key=cv2.contourArea)
    cv2.drawContours(whale, c, -1, (0, 255, 0), 2)
    # cv2.imshow("Whale", whale)

    return c

def applyMask(c, image):
    mask = np.zeros(image.shape[:2], dtype = "uint8")
    cv2.fillConvexPoly(mask, c, (255, 255, 255))
    # cv2.imshow("Mask", mask)
    maskWhale = cv2.bitwise_and(image, image, mask = mask)
    cv2.imshow("Masked Whale", maskWhale)

    return mask

def plot_histogram(image, title, mask = None):
    hists = []
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    for (chan, color) in zip (chans, colors):
        hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
        hists.append(hist)
        plt.plot(hist, color = color)
        plt.xlim([0, 256])

    plt.show()

    return hists

def blue():
    blue = cv2.imread("images/blue1.png")
    # cv2.imshow("Blue Whale", blue)
    edged = imageManipulation(blue)
    erodedFrame = morphManip(edged)
    c = contours(erodedFrame, blue)
    mask = applyMask(c, blue)

    hists = plot_histogram(blue, "Histogram for Masked Image", mask = mask)

    return hists

def fin():
    fin = cv2.imread("images/fin1.png")
    # cv2.imshow("Fin Whale", fin)
    edged = imageManipulation(fin)
    erodedFrame = morphManip(edged)
    c = contours(erodedFrame, fin)
    mask = applyMask(c, fin)

    hists = plot_histogram(fin, "Histogram for Masked Image", mask = mask)

    return hists

def humpback():
    humpback = cv2.imread("images/humpback1.png")
    # cv2.imshow("Humpback Whale", humpback)
    edged = imageManipulation(humpback)
    erodedFrame = morphManip(edged)
    c = contours(erodedFrame, humpback)
    mask = applyMask(c, humpback)

    hists = plot_histogram(humpback, "Histogram for Masked Image", mask = mask)

    return hists

def beluga():
    beluga = cv2.imread("images/beluga1.png")
    # cv2.imshow("Beluga Whale", beluga)
    edged = imageManipulation(beluga)
    erodedFrame = morphManip(edged)
    c = contours(erodedFrame, beluga)
    mask = applyMask(c, beluga)

    hists = plot_histogram(beluga, "Histogram for Masked Image", mask = mask)

    return hists

def histofOImage():
    blueHists = blue()
    finHists = fin()
    humpbackHists = humpback()
    belugaHists = beluga()
    allHists = [blueHists, finHists, humpbackHists, belugaHists]

    return allHists

def histofNImage(allHists):
    pass

def main():
    allHists = histofOImage()
    histofNImage(allHists)


main()
