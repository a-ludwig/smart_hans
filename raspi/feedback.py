import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
import socket
import time
from utils.button import Button
from gpiozero import LEDCharDisplay, LEDMultiCharDisplay

def main():
    b_right = Button(label = 'right')
    b_false = Button(label = 'false')

    button_dict = { ## dictionary with buttons and corresponding response for the socket
        b_right: True,
        b_false: False
    }   

    char = LEDCharDisplay(26, 19, 13, 6, 5, 22, 4, dp=23, active_high=False)
    #declared the GPIO pins for (a,b,c,d,e,f,g) and declared its CAS

    display = LEDMultiCharDisplay(char,25, 24, active_high=False)


    setup_gpio()
    setup_button(15, button_callback, b_right)
    setup_button(17, button_callback, b_false)

    # Define the IP address and port to listen on
    IP_ADDRESS = '0.0.0.0'  # Listen on all available interfaces
    PORT = 1234

    # Create a socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Set the SO_REUSEADDR option
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Bind the socket to the IP address and port
    s.bind((IP_ADDRESS, PORT))

    # Listen for incoming connections
    s.listen(1)

    while True:
        # Wait for a client to connect
        conn, addr = s.accept()
        conn.settimeout(0.1)
        # Receive the number from the client
        data = conn.recv(1024).decode()
        number = int(data)

        # Send an immediate response to the client
        response = "received"
        conn.sendall(response.encode())

        display.value = (str(number))
        result = False
        for button, value in button_dict.items():
            button.pushed = False  # Reset the pushed flag     
        print('waiting for button press')
        while result == False:
            for button, value in button_dict.items():
                if button.pushed:
                    print('pushed in the loop')
                    # Send the boolean value back to the client
                    response = str(value).encode()
                    conn.sendall(response)
                    result = True
                
            # Check if the socket has been closed
            try:
                data = conn.recv(1024)
            except:
                if not data:
                    print("client closed the socket")
                    result = True
            
            
                           
        display.value = ('00')
        
        # Close the connection
        conn.close()

    GPIO.cleanup() # Clean up


def setup_gpio():
    GPIO.setwarnings(False)

def setup_button(pin_number, callback_function, button):
    GPIO.setup(pin_number, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.add_event_detect(pin_number, GPIO.RISING, callback=lambda channel: callback_function(channel, button), bouncetime=200)

def button_callback(channel, button):
    button.pushed = True
    if button.label == 'right':
        print("rigth was pushed!")
    elif button.label == 'false':
        print("false was pushed!")

if __name__ == "__main__":
    main()