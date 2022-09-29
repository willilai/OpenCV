from matplotlib import pyplot as plt
from statistics import mean
import numpy as np
import argparse
import cv2

def imageManipulation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    sobelX = cv2.Sobel(blurred, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(blurred, cv2.CV_64F, 0, 1)

    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))

    sobelCombined = cv2.bitwise_or(sobelX, sobelY)
    cv2.imshow("sobelCombined", sobelCombined)
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
    # cv2.imshow("Mask", mask)
    maskWhale = cv2.bitwise_and(image, image, mask = mask)
    cv2.imshow("Masked Whale", maskWhale)
    cv2.waitKey(0)

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

    #plt.show()

    return hists

def getHistogram(name):
    whale = cv2.imread(name)
    edged = imageManipulation(whale)
    erodedFrame = morphManip(edged)
    c = contours(erodedFrame, whale)
    mask = applyMask(c, whale)

    hists = plot_histogram(whale, "Histogram for Masked Image", mask = mask)

    return hists

def histofOImage():
    blueHists = getHistogram("images/blue1.png")
    finHists = getHistogram("images/fin1.png")
    humpbackHists = getHistogram("images/humpback1.png")
    belugaHists = getHistogram("images/beluga1.png")
    allHists = [blueHists, finHists, humpbackHists, belugaHists]

    return allHists

def comparingHistograms(allHists, curHists):
    corrValues = []
    whaleSpecies = ["Blue", "Fin", "Humpback", "Beluga"]
    #for whaleHists in allHists:
    for i in range(len(allHists)):
        whaleHists = allHists[i]
        print("\n%s Whale" % whaleSpecies[i])
        whaleCorrValues = []
        for i in range(3):
            color = ["Blue", "Green", "Red"]
            colorWhaleHist = whaleHists[i]
            colorcurHist = curHists[i]

            value = cv2.compareHist(colorWhaleHist, colorcurHist, cv2.HISTCMP_CORREL)
            print(color[i] + ": " + str(value))
            whaleCorrValues.append(value)
        corrValues.append(whaleCorrValues)


    maxCorrValueWhale = [0, ""]
    for i in range(4):
        meanCorrValues = mean(corrValues[i])
        if meanCorrValues > maxCorrValueWhale[0]:
            maxCorrValueWhale[0] = meanCorrValues
            maxCorrValueWhale[1] = whaleSpecies[i]

    return maxCorrValueWhale[1]


def histofNImage(allHists):
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    args = vars(ap.parse_args())

    hists = getHistogram(args["image"])
    whaleSpecies = comparingHistograms(allHists, hists)

    print("\nIt is a %s whale. " % whaleSpecies)

def main():
    allHists = histofOImage()
    histofNImage(allHists)


main()
