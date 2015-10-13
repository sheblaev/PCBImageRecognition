# USAGE
# python sorting_contours.py 
# import the necessary packages
import numpy as np
import cv2
from matplotlib import pyplot as plt

        
def imshow(name, image):
    plt.imshow(image)
    plt.title(name)    
    plt.show()


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
        1.0, (255, 255, 255), 2)

    cv2.drawContours(image, c, -1, (0, 255,  255))
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image,[box],0,(0,255,0 ),2)
    # return the image with the contour number drawn on it
    return image

# construct the argument parser and parse the arguments
# load the image and initialize the accumulated edge image
image = cv2.imread("photo1.jpg")
gray = image
#cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imshow("image", image)

# in both the x and y direction
gray = image
#gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
#gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

# subtract the y-gradient from the x-gradient
#gradient = cv2.subtract(gradX, gradY)
#gradient = cv2.convertScaleAbs(gradient)

# blur and threshold the image
blurred = cv2.blur(gray, (5, 5))
imshow("gray", gray)
imshow("blurred", blurred)
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
closed= blurred
# construct a closing kernel and apply it to the thresholded image
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
#closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# perform a series of erosions and dilations
#closed = cv2.erode(closed, None, iterations = 4)
#closed = cv2.dilate(closed, None, iterations = 4)
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
