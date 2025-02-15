import socketio
import base64
import json
import os
import sys

# Define the SocketIO server URL
SERVER_URL = "http://127.0.0.1:8080"  # Replace with your server's address
AUDIO_FILE_PATH = sys.argv[1]  # Path to your audio file

# Initialize the SocketIO client
sio = socketio.Client()

# Event handler for connecting to the server
@sio.on("connect")
def on_connect():
    print("Connected to the server.")

# Event handler for receiving messages from the server
@sio.on("message")
def on_message(data):
    print(f"Message received from server: {data}")

# Event handler for disconnection
@sio.on("disconnect")
def on_disconnect():
    print("Disconnected from the server.")

# Function to send an audio file to the server
def send_audio(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        # Read the audio file as bytes
        with open(file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        # Encode the audio as Base64
        encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")

        # Prepare the JSON payload
        payload = {"audio": encoded_audio}

        # Send the payload to the server
        print("Sending audio to server...")
        sio.emit("message", payload)
    except Exception as e:
        print(f"Error: {e}")

# Main function
if __name__ == "__main__":
    try:
        # Connect to the server
        print("Connecting to the server...")
        sio.connect(SERVER_URL)

        # Send the audio file
        send_audio(AUDIO_FILE_PATH)

        # Wait for a response from the server
        sio.wait()
    except Exception as e:
        print(f"Error: {e}")
