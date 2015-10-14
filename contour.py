# USAGE
# python sorting_contours.py 
# import the necessary packages
import numpy as np
import cv2
from matplotlib import pyplot as plt

RED = (255,0,0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0,0,0)

def imshow(name, image):
    plt.imshow(image)
    plt.title(name)    
    plt.show()
    cv2.imshow(name, image)
    cv2.waitKey()

def draw_contour(image, c, i):
    # compute the center of the contour area and draw a circle
    # representing the center
    M = cv2.moments(c)
    if M["m00"] == 0:
        return

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # draw the countour number on the image
    cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, GREEN, 2)

    cv2.drawContours(image, c, -1, (0, 255,  255))
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image,[box],0,(GREEN),2)
    # return the image with the contour number drawn on it
    return image

# construct the argument parser and parse the arguments
# load the image and initialize the accumulated edge image
image = cv2.imread("photo1_dropped.jpg")
#cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imshow("image", image)

# in both the x and y direction

# perform a series of erosions and dilations
for i in range(1,5) :
	closed = cv2.erode(image, None, iterations = 2)
	closed = cv2.dilate(closed, None, iterations = 2)

imshow("closed", closed)
image = closed

accumEdged = np.zeros(image.shape[:2], dtype="uint8")

# loop over the blue, green, and red channels, respectively
for chan in cv2.split(image):
    # blur the channel, extract edges from it, and accumulate the set
    # of edges for the image
    chan = cv2.medianBlur(chan, 11)
    imshow("chan", chan)

edged = cv2.Canny(closed, 150, 250)
accumEdged = cv2.bitwise_or(accumEdged, edged)

# show the accumulated edge map
imshow("Edge Map", accumEdged)

# find contours in the accumulated image, keeping only the largest
# ones
(_, cnts, _) = cv2.findContours(accumEdged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5:]
orig = image.copy()

# loop over the (unsorted) contours and draw them
for (i, c) in enumerate(cnts):
    orig = draw_contour(orig, c, i)

# show the original, unsorted contour image
imshow("Unsorted", orig)

# sort the contours according to the provided method
#(cnts, boundingBoxes) = sort_contours(cnts, method=args["method"])

# loop over the (now sorted) contours and draw them
for (i, c) in enumerate(cnts):
    draw_contour(image, c, i)

# show the output image
imshow("Sorted", image)
cv2.waitKey(0)
