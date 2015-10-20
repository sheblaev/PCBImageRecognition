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

def invert_dict_nonunique(d):
    newdict = {}
    for k, v in d.iteritems():
        newdict.setdefault(v, []).append(k)
    return newdict

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
image = cv2.imread("photo3.jpg")
imshow("initial", image)
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(_, w, h) = image .shape[::-1]
print (w, h)

value = {[0,0,0]:0}
# Let's detect the most frequents color on board and remove them before grayscaling
for x  in range(w):
        for y in range(h):
                c =  image[y][x]
                if not c in value: 
                        value[c] = 1
                else: 
                        value[c] +=1

inverted = invert_dict_nonunique(value)

del_colors =  sorted(inverted)[-40:]

for i in range(len(del_colors)):
        print i, inverted[del_colors[i]]

imshow("gray", gray)
print del_colors
for x  in range(w):
        for y in range(h):
                if value [gray[y][x] ] in del_colors:
                       gray[y][x] = 0 


gray = cv2.medianBlur(gray, 3)

accumEdged = np.zeros(image.shape[:2], dtype="uint8")
edged = cv2.Canny(gray, 150, 250)
accumEdged = cv2.bitwise_or(accumEdged, edged)

(_, cnts, _) = cv2.findContours(accumEdged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:15:]
for (i, c) in enumerate(cnts):
    draw_contour(image, c, i)



imshow("filtered", gray)
imshow("image", image)
imshow("image", accumEdged)
cv2.imwrite("gray.png", gray)
cv2.waitKey()
