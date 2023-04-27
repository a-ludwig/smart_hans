import socket
from utils.helper import *
from utils.horse import horse

# Define the IP address and port of the Raspberry Pi
IP_ADDRESS = '192.168.1.195'  # Replace with the IP address of your Raspberry Pi
PORT = 1234
PI_RESPONSE_TIME = 7

hansi = horse(max_taps=12) ##hansi is our magnificent horse 
hansi.queue()## keep hansi alive

feedback = send_rec_feedback(IP_ADDRESS, PORT, hansi, PI_RESPONSE_TIME)
print("ANSWER_________"+str(feedback))