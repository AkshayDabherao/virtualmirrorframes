import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import sys

if len(sys.argv) > 1:
        address = sys.argv[1]
        print 'loading %s ...' % address
        img = cv2.imread(address)
        if img is None:
            print 'Failed to load address:', address
            sys.exit(1)
else:
    	address = './data/eye01.jpg'

img = cv2.imread(address)
face_cascade = cv2.CascadeClassifier('../opencv-3.0.0/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../opencv-3.0.0/data/haarcascades/haarcascade_eye.xml')

def detectFaces(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=4)
	for (x,y,w,h) in faces:
		cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		
	eyes = eye_cascade.detectMultiScale(gray)
	for (a,b,c,d) in eyes:
		cv2.rectangle(gray,(a,b),(a+c,b+d),(255,0,0),2)
		roi_eye = gray[b:b+d,a:a+c]
	return gray#,roi_eye


ans  = detectFaces(img)
#plt.subplot(121),plt.imshow(lol,'gray'),plt.title('myharris')
plt.subplot(122),plt.imshow(ans,'gray'),plt.title('inbuilt harris')
plt.show()