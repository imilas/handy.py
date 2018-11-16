import numpy as np
import cv2
import copy
import math
import random
import argparse

# Import needed modules from osc4py3
from osc4py3.as_eventloop import *
from osc4py3 import oscbuildparse
import time

# parameters
cap_region_x_end=0.5  # start point/total width
cap_region_y_end=0.5  # start point/total width
threshold = 20 #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# variables
captured = False   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works
#send osc messages:
osc_startup()

# Make client channels to send packets.
osc_udp_client("0.0.0.0", 6448, "aclientname")


def printThreshold(thr):
    return

# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)
# cv2.namedWindow('tSlider')
# cv2.createTrackbar('threshold', 'tSlider', threshold, 100, printThreshold)

def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


while camera.isOpened():
    # keyboard inputs
    k = cv2.waitKey(10)
    if k == 27 or k == ord('q'):  # q or esc to exit
        # cv2.imwrite("last.jpg", im2) 
        # Properly close the system.
        osc_terminate()
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        captured = True

    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)

    cv2.rectangle(frame, (0, 0),
                 (int(frame.shape[1]*0.5), int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)


    if captured:

        # convert image to binary image
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                    0:int(frame.shape[1]*0.5)]  # clip the ROI        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

        #find contours then get the top x largest ones
        im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        contours.sort(key=lambda cnt: cv2.contourArea(cnt))
        contours=contours[-1:]

        color=( random.randint(100,200), random.randint(100,200), random.randint(100,200));
            
        cv2.drawContours(img, contours, -1, color, random.randint(5,10))

        cv2.imshow('binaried & contoured', img)

        #find some (7?) number of centroids for each contour
        for cont in contours:
            
            try:
               
                hull = cv2.convexHull(cont)
                clusters = 7
                hull = np.squeeze(hull)
                if np.float32(hull).size>(clusters * 2):  
                    #kmeans 
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
                    ret,label,center=cv2.kmeans( np.float32(hull),clusters,None,criteria,3,cv2.KMEANS_PP_CENTERS)

                    for px in np.squeeze(np.int32([center])  ):

                        px = tuple(px)
                        cv2.circle(img,px, 0, (99,12,255), 20)
                    
                    # Build and send the OSC message 
                    wekInputs = list(center.flatten()) 
                    msg = oscbuildparse.OSCMessage("/wek/inputs", None, [float(i) for i in wekInputs ])
                    osc_send(msg, "aclientname")
                    osc_process()
            except IndexError:
                print("waiting for your moves")
            
