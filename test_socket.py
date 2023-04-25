import socket

# Define the IP address and port of the Raspberry Pi
IP_ADDRESS = '192.168.1.195'  # Replace with the IP address of your Raspberry Pi
PORT = 1234

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the Raspberry Pi
s.connect((IP_ADDRESS, PORT))

# Send a number to the Raspberry Pi
number = 42
s.sendall(str(number).encode())

# Receive the response from the Raspberry Pi
response = s.recv(1024).decode()
result = response

# Close the socket
s.close()

# Print the result
print(f"The boolean value is {result}")