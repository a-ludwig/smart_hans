from gpiozero import LEDCharDisplay, LEDMultiCharDisplay
import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
#import the LEDCharDisplay library from gpiozero
from time import sleep
#import the sleep library from time

char = LEDCharDisplay(26, 19, 13, 6, 5, 22, 4, dp=23, active_high=False)
#declared the GPIO pins for (a,b,c,d,e,f,g) and declared its CAS

display = LEDMultiCharDisplay(char,25, 24, active_high=False)


display.value = ('HI')

pushed = False
def button_callback(channel):
    global pushed
    pushed = True
GPIO.setwarnings(False) # Ignore warning for now
GPIO.setup(15, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 10 to be an input pin and set initial value to be pulled low (off)
GPIO.add_event_detect(15,GPIO.RISING,callback=button_callback) # Setup event on pin 10 rising edge

while not pushed:
    pass
#initialize the infinite while loop
    
    # for char in '0123456789':
    #  #initialize for loop and store 0123456789 in variable char

    #      display.value = char
    #      #displayed the value

    #      sleep(1)
    #      #generated delay of one second
display.value = ('BY')
sleep(2)