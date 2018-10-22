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

stream_bytes = b''        
while True:
    rc_car = RCControl()
    stream_bytes += stream.read(1024)
    first = stream_bytes.find(b'\xff\xd8')
    last = stream_bytes.find(b'\xff\xd9')
    if first != -1 and last != -1:
        jpg = stream_bytes[first:last+2]
        stream_bytes = stream_bytes[last+2:]
        gray = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

                    # lower half of the image
        half_gray = gray[120:240, :]
        cv2.imshow('image', gray)

        #cv2.imshow('half', half_gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
           rc_car.stop()
           break
        image_array = half_gray.reshape(1, 38400).astype(np.float32)
                    #predict
        a, b = model.predict(image_array)
        prediction = b.argmax(-1)
        print( prediction)
                    #drive
        rc_car.steer(prediction)
        
stream.close()

        

cv2.destroyAllWindows()

