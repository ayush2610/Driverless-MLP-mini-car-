import cv2
import numpy as np
import math
import bluetooth
import urllib.request
print( "Searching for devices...")
print( "")

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

stream=urllib.request.urlopen('http://172.16.45.127:8080/video')	
#create a model
model = cv2.ml.ANN_MLP_load('mlp_xml/mlp.xml')
#predict

class RCControl(object):

    def steer(self, prediction):
        if prediction == 2:
            sock.send('F')
            print("Forward")
        elif prediction == 1:
            sock.send('R')
            print("Right")
        elif prediction == 0:
            sock.send('L')
            print("Left")
        elif prediction == 3:
            sock.send('S')
            print('Stop')
        else:
            sock.send('S')

    def stop(self):
        sock.send('S')

class ObjectDetection(object):

    def __init__(self):
        self.red_light = False
        self.green_light = False
        self.yellow_light = False
        self.light=False
        self.stopsign=False
        self.ped=False

    def detect(self, cascade_classifier, gray_image, image,scaleFactor,minNeighbors,num ):
        v = 0
        lower = [0]
        upper = [80]
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")        
        cascade_obj = cascade_classifier.detectMultiScale(
            gray_image,
            scaleFactor,
            minNeighbors
        )
        for (x_pos, y_pos, width, height) in cascade_obj:
            cv2.rectangle(image, (x_pos+5, y_pos+5), (x_pos+width-5, y_pos+height-5), (255, 255, 255), 2)
            v = y_pos + height - 5
            if  num==1:
                cv2.putText(image, 'STOP', (x_pos, y_pos-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.stopsign=True
            elif num==2:
                roi = gray_image[y_pos+10:y_pos + height-10, x_pos+10:x_pos + width-10]
                mask = cv2.inRange(roi, lower, upper)
                output = cv2.bitwise_and(roi, roi, mask=mask)
              
                ret,thresh = cv2.threshold(mask, 10, 255, 0)
                im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if len(contours) != 0:
                   c = max(contours, key = cv2.contourArea)
                   msk2=np.zeros(roi.shape, dtype='uint8')
                   al=cv2.drawContours(msk2, [c], 0, (255,255,255), -1)
                   onea=cv2.bitwise_and(roi,al)
                   cv2.imshow('one',onea)
                   #grayal=cv2.cvtColor(onea,cv2.COLOR_BGR2GRAY)
                   (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(onea)
                   if maxVal - minVal > 5:
                   	cv2.circle(image, (x_pos+maxLoc[0],y_pos+maxLoc[1]), 5, (255, 255, 0), 2)
                   	x,y,w,h = cv2.boundingRect(c)
                   	if maxLoc[1]<(y+h)/3:
                                print('red')    
                                cv2.putText(image, 'Red', (x_pos+5, y_pos-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                self.red_light = True
                                self.light=True
                   	elif maxLoc[1]>(y+h)*2/3:
                                print('green')
                                cv2.putText(image, 'Green', (x_pos+5, y_pos-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                self.green_light = True
                                self.light=True
                   	else :
                     		print('none')
            elif num==3:
                        cv2.putText(image, 'Pedestrian', (x_pos, y_pos-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        self.ped=True
                        
        return v


stream_bytes = b''       
obj_detection = ObjectDetection() 
stop_cascade = cv2.CascadeClassifier('stopsign.xml')
light_cascade = cv2.CascadeClassifier('traf.xml')
ped_cascade = cv2.CascadeClassifier('haarcascade_pedestrian.xml')
stop_flag = False
stop_sign_active = False
while True:
    obj_detection.stopsign=False
    obj_detection.light=False
    obj_detection.ped=False
    rc_car = RCControl()
    stream_bytes += stream.read(1024)
    first = stream_bytes.find(b'\xff\xd8')
    last = stream_bytes.find(b'\xff\xd9')
    stop_flag = False
    stop_sign_active = False
    if first != -1 and last != -1:
        jpg = stream_bytes[first:last+2]
        stream_bytes = stream_bytes[last+2:]
        gray = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), 0)
        image= cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), 1)
        half_gray = gray[120:240, :]
        cv2.imshow('image', gray)
        cv2.imshow('half', half_gray)

        v_param1 = obj_detection.detect(stop_cascade, gray, image,scaleFactor=1.5,minNeighbors=25,num=1)
        v_param2 = obj_detection.detect(light_cascade, gray, image,scaleFactor=1.5,minNeighbors=15,num=2)
        v3 = obj_detection.detect(ped_cascade,gray, image,scaleFactor=1.7, minNeighbors=20,num=3)
        image_array = half_gray.reshape(1, 38400).astype(np.float32)
        a, b = model.predict(image_array)
        prediction = b.argmax(-1)
        if stop_sign_active and obj_detection.stopsign:
                     print("Stop sign ahead")
                     stop()

        if stop_flag is False:
              stop_start = cv2.getTickCount()
              stop_flag = True
              stop_finish = cv2.getTickCount()

              stop_time = (stop_finish - stop_start)/cv2.getTickFrequency()
              print( "Stop time: %.2fs" % stop_time)
              if stop_time > 5:
                   print("Waited for 5 seconds")
                   stop_flag = False
                   stop_sign_active = False
              elif obj_detection.light:

                        if obj_detection.red_light:
                            print("Red light")
                            stop()
                        elif obj_detection.green_light:
                            print("Green light")
                            pass
                        elif obj_detection.yellow_light:
                            print("Yellow light flashing")
                            pass
                        
                        
                        obj_detection.red_light = False
                        obj_detection.green_light = False
                        obj_detection.yellow_light = False
              elif self.obj_detection.ped:
                         print('pedestrian ahead')
                         rc_car.stop()
              else:
                        self.steer(prediction)
                        self.stop_start = cv2.getTickCount()
                        self.obj_detection.stopsign=False
                        
                        if stop_sign_active is False:
                            self.drive_time_after_stop = (self.stop_start - self.stop_finish)/cv2.getTickFrequency()
                            if self.drive_time_after_stop > 5:
                                stop_sign_active = True
              if cv2.waitKey(1) & 0xFF == ord('q'):
                        rc_car.stop()
                        break

            cv2.destroyAllWindows()

        
stream.close()
cv2.destroyAllWindows()

