from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the Image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (21, 21), 0)
cv2.imshow("Image", image)
sobelX = cv2.Sobel(blurred, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(blurred, cv2.CV_64F, 0, 1)

sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

sobelCombined = cv2.bitwise_or(sobelX, sobelY)

edged = cv2.Canny(sobelCombined, 10, 70)
cv2.imshow("Edges", edged)

 #morphology ops
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23,23))
kernel2 = np.ones((15,15), np.uint8)
#eliplical erode to get rid of some of the noise
openFrame = cv2.dilate(edged, kernel2, iterations = 1)
erodedFrame = cv2.erode(openFrame, kernel1, iterations = 1)
cv2.imshow("morhp ops", erodedFrame)

(cnts, _) = cv2.findContours(erodedFrame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

whale = image.copy()
c = max(cnts, key=cv2.contourArea)
cv2.drawContours(whale, c, -1, (0, 255, 0), 2)
cv2.imshow("Whale", whale)

"""
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)

    whale = image[y:y + h, x:x+w]
    cv2.imshow("Whale", whale)

    mask = np.zeros(image.shape[:2], dtype = "uint8")
    ((centerX, centerY), radius) = cv2.minEnclosingCircle(c)

    cv2.circle(mask, (int(centerX), int(centerY)), int(radius), 255, -1)
    mask = mask[y:y + h, x:x + w]
    cv2.imshow("Masked Coin", cv2.bitwise_and(coin, coin, mask = mask))
    cv2.waitKey(0)

cv2.waitKey(0)
""""""
#HISTOGRAMS
chans = cv2.split(image)
colors = ("b", "g", "r")
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [200, 256])
    plt.plot(hist, color = color)
    plt.xlim([0, 256])

fig = plt.figure()

ax = fig.add_subplot(131)
hist = cv2.calcHist(chans[0], [0], None, [32], [200, 256])
p = ax.imshow(hist, interpolation = "nearest")
ax.set_title("Color Histogram for B")
plt.plot(hist)

hist = cv2.calcHist(chans[1], [0], None, [32], [200, 256])
p = ax.imshow(hist, interpolation = "nearest")
ax.set_title("2D Color Histogram for G")
plt.plot(hist)

ax = fig.add_subplot(133)
hist = cv2.calcHist(chans[2], [0], None, [32], [200, 256])
p = ax.imshow(hist, interpolation = "nearest")
ax.set_title("2D color Histogram for R")
plt.plot(hist)
print("2D histogram shape: {}, with {} values".format(hist.shape, hist.flatten().shape[0]))

plt.show()
"""
cv2.waitKey(0)
