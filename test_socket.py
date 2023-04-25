import socket
from utils.helper import *

# Define the IP address and port of the Raspberry Pi
IP_ADDRESS = '192.168.1.195'  # Replace with the IP address of your Raspberry Pi
PORT = 1234

s = connect_socket(IP_ADDRESS, PORT)

feedback = get_feedback(s, 20)