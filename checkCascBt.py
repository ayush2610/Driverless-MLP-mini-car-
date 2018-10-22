__author__ = 'zhengwang'

import numpy as np
import cv2
import pygame
from pygame.locals import *
import socket
import time
import os
import urllib.request

pb_cascade = cv2.CascadeClassifier('cascades/stopsign.xml')
def detect(frame):
    cases = pb_cascade.detectMultiScale(frame, 1.5 ,10) #1.5 20 for stopsign#1.7 3 for cars video
    pxwidth = 0
    for (x,y,w,h) in cases:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    return frame

stream=urllib.request.urlopen('http://192.168.43.1:8080/video')

try:
            stream_bytes = b' '
            while True:
                stream_bytes += stream.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')

                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),0)
                    cv2.imshow('image', image)
                    #cv2.imshow('roi',roi)
                    new = detect(image)
                    cv2.imshow('nw',new)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                         break
    

finally:
            cv2.destroyAllWindows()

