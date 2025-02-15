import socket
import os
import requests

class NetworkManager:
    def __init__(self):
        self.server_ip = "169.254.58.36"
        # self.server_ip = "10.181.88.49"
        self.server_port = 12345
        self.url = "https://43ef-91-73-80-240.ngrok-free.app/udp"
        #self.url = "http://127.0.0.1:5000/udp"
    
    def send_message(self, message):
        try:
            server_address = (self.server_ip, self.server_port)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client_socket:
                # Send the message to the server
                print(message)
                client_socket.sendto(message.encode(), server_address)
                # Receive response from the server
            # client_socket.close()
        except Exception as e:
            print(f"Error: Sending udp message: {e}")

    def send_http(self, message):
        data = {"udp":message}
        response = requests.post(self.url, json=data)
        if response.status_code != 200:
            print(response)