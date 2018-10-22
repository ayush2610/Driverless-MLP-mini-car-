import numpy as np
import cv2

import pygame
from pygame.locals import *
import socket
import time
import os
import bluetooth
import urllib.request
print("Searching for devices...")
print ("")

nearby_devices = bluetooth.discover_devices()
#Run through all the devices found and list their name
num = 0
print ("Select your device by entering its coresponding number...")
for i in nearby_devices:
	num+=1
	print( num , ": " , bluetooth.lookup_name( i ))

#Allow the user to select their Arduino
#bluetooth module. They must have paired
#it before hand.
selection = int(input("> ")) - 1
print( "You have selected", bluetooth.lookup_name(nearby_devices[selection]))
bd_addr = nearby_devices[selection]

port = 1
sock = bluetooth.BluetoothSocket( bluetooth.RFCOMM )
sock.connect((bd_addr, port))
stream = urllib.request.urlopen('http://192.168.43.1:8080/video')
class CollectTrainingData(object):
    def __init__(self):

        # create labels
        self.k = np.zeros((4, 4), 'float')
        for i in range(4):
            self.k[i, i] = 1
        self.temp_label = np.zeros((1, 4), 'float')
        self.send_inst = True
        pygame.init()
        screen = pygame.display.set_mode((340, 240))
        self.collect_image()
        sock = bluetooth.BluetoothSocket( bluetooth.RFCOMM )        
    def collect_image(self):

        saved_frame = 0
        total_frame = 0

        # collect images for training
        print( 'Start collecting images...')
        e1 = cv2.getTickCount()
        image_array = np.zeros((1, 38400))
        label_array = np.zeros((1, 4), 'float')
        stream_bytes = b''
        frame = 1
            
        while self.send_inst:
            stream_bytes += stream.read(1024)
            first = stream_bytes.find(b'\xff\xd8')
            last = stream_bytes.find(b'\xff\xd9')
            if first != -1 and last != -1:
                jpg = stream_bytes[first:last + 2]
                stream_bytes = stream_bytes[last + 2:]
                image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    # select lower half of the image
                roi = image[120:240,:]
            
                    # save streamed images
                cv2.imwrite('training_images/frame{:>05}.jpg'.format(frame), image)
                cv2.imshow('image', image)
                cv2.imshow('roi', roi)
                if cv2.waitKey(1) ==27:
                    exit(0)
    
                    # reshape the roi image into one row array
                temp_array = roi.reshape(1, 38400).astype(np.float32)
                    
                frame += 1
                total_frame += 1
                    #print   saved_frame

                    # get input from human driver
                for event in pygame.event.get():
                    if event.type == KEYDOWN:
                        key_input = pygame.key.get_pressed()
                            # complex orders
                        if key_input[pygame.K_UP]:
                            print("Forward")
                            saved_frame += 1
                            image_array = np.vstack((image_array, temp_array))
                            label_array = np.vstack((label_array, self.k[2]))
                            sock.send("F")
                            
                        elif key_input[pygame.K_RIGHT]:
                            print("Right")
                            image_array = np.vstack((image_array, temp_array))
                            label_array = np.vstack((label_array, self.k[1]))
                            saved_frame += 1
                            sock.send("R")

                        elif key_input[pygame.K_LEFT]:
                            print("Left")
                            image_array = np.vstack((image_array, temp_array))
                            label_array = np.vstack((label_array, self.k[0]))
                            saved_frame += 1
                            sock.send("L")

                        elif key_input[pygame.K_q]:
                            print( 'exitSTOP')
                            image_array = np.vstack((image_array, temp_array))
                            label_array = np.vstack((label_array, self.k[3]))
                            saved_frame += 1
                            self.send_inst = False
                            sock.send("S")
                            break
                                    
                    elif event.type == pygame.KEYUP:
                        print ("keyup")
                        sock.send("S")

            # save training images and labels
        train = image_array[1:, :]
        train_labels = label_array[1:, :]

            # save training data as a numpy file
        file_name = str(int(time.time()))
        print( file_name)
        directory = "training_data"
        if not os.path.exists(directory):
            os.makedirs(directory)
        try:    
            np.savez(directory + '/' + file_name + '.npz', train=train, train_labels=train_labels)
        except IOError as e:
            print(e)

        e2 = cv2.getTickCount()
            # calculate streaming duration
        time0 = (e2 - e1) / cv2.getTickFrequency()
        print( 'Streaming duration:', time0)

        print(train.shape)
        print(train_labels.shape)
        print( 'Total frame:', total_frame)
        print( 'Saved frame:', saved_frame)
        print ('Dropped frame', total_frame - saved_frame)

if __name__ == '__main__':
    
    CollectTrainingData()
