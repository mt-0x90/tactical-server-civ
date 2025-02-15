from network import NetworkManager
import pygame
import sys
import time 
pygame.mixer.init()

def play_audio(file_path):
    """
    Load and play an audio file.
    Prints "start" immediately after playback begins.
    """
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        print("start")
    except pygame.error as e:
        print(f"Error loading/playing audio: {e}")

def monitor_audio():
    """
    Wait until the audio finishes playing naturally,
    then print "stop".
    """
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    print("stop")

def stop_audio_if_needed(param):
    """
    If the passed parameter is 1, stop the audio if it's still playing,
    and print "stop".
    """
    if param == 1:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            print("stop")


fpath = sys.argv[1]
play_audio(fpath)
time.sleep(1)  # Adjust time as needed
# If parameter is 1, this will stop the audio immediately and print "stop".
stop_audio_if_needed(1)