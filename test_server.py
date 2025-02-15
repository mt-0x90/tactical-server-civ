from flask import Flask
from flask_socketio import SocketIO
import base64


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on("connect")
def handle_connect():
    print("Client connected!")

@socketio.on("message")
def handle_message(data):
    print("Message received from client!")

    # Check if 'audio' key is in the received data
    if isinstance(data, dict) and "audio" in data:
        try:
            # Decode the Base64-encoded audio
            encoded_audio = data["audio"]
            audio_bytes = base64.b64decode(encoded_audio)

            # Save the received audio to a file
            with open("received_audio.wav", "wb") as audio_file:
                audio_file.write(audio_bytes)
            print("Audio file saved as 'received_audio.mp3'.")

            # Send a response back to the client
            response = {"response": "Audio processed successfully!"}
            socketio.send(response)
        except Exception as e:
            print(f"Error processing audio: {e}")
            socketio.send({"response": "Error processing audio."})
    else:
        print("Invalid message received. No 'audio' key found.")
        socketio.send({"response": "Invalid message format."})

@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected.")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8080, debug=True)
