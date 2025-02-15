import socket
import os

class NetworkManager:
    def __init__(self):
        self.server_ip = "localhost"
        self.server_port = 1235

    def setup_dummy_server(self):
        try:
            server_address = (self.server_ip, self.server_port)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server_socket:
                # Bind the socket to the address and port
                server_socket.bind(server_address)
                print(f"Server listening on {server_address[0]}:{server_address[1]}")
                while True:
                    # Receive data from the client
                    data, client_address = server_socket.recvfrom(1024)
                    print(f"Received message from {client_address}: {data.decode()}")
                    # Send a response back to the client
                    response = f"Server received: {data.decode()}"
                    server_socket.sendto(response.encode(), client_address)
        except Exception as e:
            print(f"Exception occured: {e}")

netx = NetworkManager()
netx.setup_dummy_server()