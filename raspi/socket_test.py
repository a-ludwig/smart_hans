import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
import socket
import time
pushed = False
def button_callback(channel):
    global pushed
    pushed = True
    print("Button was pushed!")
GPIO.setwarnings(False) # Ignore warning for now
GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 10 to be an input pin and set initial value to be pulled low (off)
GPIO.add_event_detect(10,GPIO.RISING,callback=button_callback) # Setup event on pin 10 rising edge
#message = input("Press enter to quit\n\n") # Run until someone presses enter




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
    
    # Receive the number from the client
    data = conn.recv(1024).decode()
    number = int(data)
    
    print("waiting")
    time.sleep(5)
    print("response")
    # Process the number and generate a boolean value
    result = pushed
    pushed = False  # Reset the pushed flag
    print(result)
    print(f"was the number:{number} correct?")
    # Send the boolean value back to the client
    response = str(result).encode()
    conn.sendall(response)
    
    # Close the connection
    conn.close()

GPIO.cleanup() # Clean up