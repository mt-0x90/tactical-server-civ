from flask import Flask
from flask_socketio import SocketIO
from rag import RagPipeline
import pygame
from pathlib import Path
from network import NetworkManager
from emailHandler import EmailHandler
import threading
import time
import eventlet
from playsound import playsound
from sttHandler import SttPipeline
import base64
import json
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")
pygame.mixer.init()
# email_handler = EmailHandler("credentials.json")
report_fpath = "docs/sample.pdf"
creds = json.load(open(os.environ['AGENTCONFIG']))
stt_handler = SttPipeline(config=creds, use_local=True)
rag = RagPipeline(use_ollama=True)
network_manager = NetworkManager()

def play_audio(fpath):
    playsound(fpath)

def play_audio(file_path):
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        network_manager.send_message("start")
        while pygame.mixer.music.get_busy():
            eventlet.sleep(0.1)  # Yield control for event processing.
        network_manager.send_message("stop")
    except pygame.error as e:
        print(f"Error loading/playing audio: {e}")

def monitor_audio():
    try:
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        network_manager.send_message("stop")
    except Exception as e:
        print(f"Exception with pygame: {e}")

his_dir = Path("history")

def get_encoded_audio(fpath):
    with open(fpath, "rb") as mp3_file:
        mp3_bytes = mp3_file.read()
    base64_string = base64.b64encode(mp3_bytes).decode('utf-8')
    return base64_string

def save_history(data):
    his_fpath = os.path.join(his_dir, "history.json")
    if os.path.exists(his_fpath):
        old = json.load(open(his_fpath))
        old.append(data)
    else:
        with open(his_fpath, 'w') as f:
            json.dump([data], f)
        f.close()
    

@socketio.on("connect")
def handle_connect():
    print("Client connected!")

@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected.")


@socketio.on("message")
def handle_audio(data):
    has_alert = False
    try:
        # Extract the flag
        flag = data["flag"]
        if flag != "reset" and flag != "dismiss":
            # Receive the scene_txt
            scene_txt = data["scene_txt"]
            # audio player
            audio_player = "server"
            # Receive the scene_type
            scene_type = data["scene_type"]
            # Email address
            email_address = data["email_address"]
            # Receive the base64-encoded audio file
            audio_file = "null"
            if scene_txt == "disable":
                audio_data = data["audio"]
                audio_bytes = base64.b64decode(audio_data)
                # Save the audio to a file for processing
                audio_file = stt_handler.fileUtils.get_fpath()
                with open(audio_file, "wb") as f:
                    f.write(audio_bytes)
                user_text = stt_handler.get_text(audio_file, is_wav=True)
            else:
                user_text = scene_txt
            # Simulate audio processing (e.g., speech-to-text, etc.)
            answer, udp_signal, is_scene = rag.handle_user_text(user_text=user_text,voice_id="matilda", scene_type=scene_type)
            base64_string = get_encoded_audio(answer)
            if scene_type == "civil" and udp_signal == "1.11":
                has_alert = True
                alert_msg = "audios/matilda/1_18747.mp3"
                response = {"response": "success", "answer":base64_string, "udp":udp_signal, "alert_base64":get_encoded_audio(alert_msg)}
            elif scene_type == "military" and udp_signal == "1.19":
                has_alert = True
                alert_msg = "audios/matilda/1_16346.mp3"
                response = {"response": "success", "answer":base64_string, "udp":udp_signal, "alert_base64":get_encoded_audio(alert_msg)}
            else:
                alert_msg = "na"
                response = {"response": "success", "answer":base64_string, "udp":udp_signal, "alert_base64":"na"}
            save_history({"user_audio":audio_file, "model_prediction":user_text, "answer":answer, "is_scene": is_scene, "scene_type":scene_type})
            response = {"response": "success", "answer":base64_string, "udp":udp_signal}
            socketio.send(response)
            network_manager.send_message(udp_signal)
            time.sleep(1)
            if audio_player == "server":
                thread = threading.Thread(target=play_audio, args=(answer,))
                thread.start()
                
                if has_alert and os.path.exists(alert_msg):
                    thread = threading.Thread(target=play_audio, args=(alert_msg,))
                    thread.start()
                    # network_manager.send_message("stop")
            if str(udp_signal) == "1.13":
                # send an email with the report to Khalid
                pass
        elif flag == "dismiss":
            # dismiss the notification
            network_manager.send_message(udp_signal)
        else:
            # reset the platform
            network_manager.send_message("1.1")
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                time.sleep(0.2)
                network_manager.send_message("stop")
    except Exception as e:
        print(f"Error occured: {e}")
        socketio.send({'response':"error in processing audio"})

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8080, debug=True)