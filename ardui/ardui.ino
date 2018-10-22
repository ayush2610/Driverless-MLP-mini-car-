#include <SoftwareSerial.h>
const int rxpin = 0;
const int txpin = 1;
char k = 'A';
//Connect the Bluetooth module
SoftwareSerial bluetooth(rxpin, txpin);
const int trigPin = 12;
const int echoPin = 13;
long duration;
int distance
void setup()
{
  //Set the lightbulb pin to put power out
  pinMode(13, OUTPUT);
  pinMode(3,OUTPUT);
  pinMode(4,OUTPUT);
  pinMode(5,OUTPUT);
  pinMode(6,OUTPUT);
  pinMode(trigPin, OUTPUT); // Sets the trigPin as an Output
  pinMode(echoPin, INPUT); // Sets the echoPin as an Input
  //Initialize Serial for debugging purposes
  Serial.begin(9600);
  Serial.println("Serial ready");
  //Initialize the bluetooth
  bluetooth.begin(9600);
  bluetooth.println("Bluetooth ready");
}

void loop()
{
  //Check for new data
  if(bluetooth.available()){
    //Remember new data
    k = bluetooth.read();
    //Print the data for debugging purposes
    Serial.println(k);
  }
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
// Sets the trigPin on HIGH state for 10 micro seconds
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
// Reads the echoPin, returns the sound wave travel time in microseconds
  duration = pulseIn(echoPin, HIGH);
// Calculating the distance
  distance= duration*0.034/2;
// Prints the distance on the Serial Monitor
  Serial.print("Distance: ");
  Serial.println(distance);
if(distance<10){
    digitalWrite(3,HIGH);
    digitalWrite(4,LOW);
    digitalWrite(5,HIGH);     
    digitalWrite(6,LOW)
}
else{
  //Turn on the light if transmitted data is H
  if( k == 'F' ){
    digitalWrite(3,HIGH);
    digitalWrite(4,LOW);
    digitalWrite(5,HIGH);     
    digitalWrite(6,LOW);
    digitalWrite(13, HIGH);
  }
  //Turn off the light if transmitted data is L
  else if( k == 'L') {
    digitalWrite(13, LOW);
    digitalWrite(3, HIGH);
    digitalWrite(4, LOW);
    digitalWrite(5, LOW);
    digitalWrite(6, LOW);
  }
  else if( k == 'R') {
    digitalWrite(13, LOW);
    digitalWrite(3, LOW);
    digitalWrite(4, LOW);
    digitalWrite(5, HIGH);
    digitalWrite(6, LOW);
  }
  else if( k == 'S') {
    digitalWrite(13, LOW);
    digitalWrite(3, LOW);
    digitalWrite(4, LOW);
    digitalWrite(5, LOW);
    digitalWrite(6, LOW);
  }
  
  //Wait ten milliseconds to decrease unnecessary hardware strain
   delay(10);
}
}
