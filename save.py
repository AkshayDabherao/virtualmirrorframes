#!/usr/bin/python

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
out = cv2.VideoWriter('output.avi',fourcc, 10.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        #frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
        
length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

cap.release()
out.release()







# Release everything if job is finished
cap2 = cv2.VideoCapture('output.avi')
ip = cv2.imread('ip3.jpg',0)
h,w = ip.shape[:2]
fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
out2 = cv2.VideoWriter('outprocessed.avi',fourcc, 10.0, (640,480))

while(cap2.isOpened()):
    ret2,img = cap2.read()
    if ret2==True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for x in range(0,h):
            for y in range(0,w):
                if ip[x,y] < 220:
                    img[x+150,y+150] = ip[x,y]
        out2.write(img)
        
        cv2.imshow('frame',img)
        cv2.waitKey(1)
    else:
        break

cap2.release()
out2.release()

cv2.destroyAllWindows()