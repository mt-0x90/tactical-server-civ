import sys
import os
import argparse
from rag import RagPipeline
from sttHandler import SttPipeline
import pandas as pd
from tqdm import tqdm
from network import NetworkManager
from playsound import playsound
from teacher import Teacher
import json
import time
import os

rag = RagPipeline()
network_manager = NetworkManager()
qa_eval = []
civilian_audio_list = ['scenario-1-question_2.mp3',
          'scenario-1-question_4.mp3',
          'scenario-1-question_3.mp3',
          'scenario-1-question_5.mp3',
          'scenario-1-question_6.mp3',
          'scenario-1-question_7.mp3',
          'scenario-1-question_8.mp3',
          'scenario-1-question_9.mp3',
          'scenario-1-question_10.mp3',
          'scenario-1-question_11.mp3',
          'scenario-1-question_12.mp3',
          'scenario-1-question_13.mp3',
          'scenario-1-question_14.mp3',
          'scenario-1-question_15.mp3',
          'scenario-1-question_16.mp3',
          'scenario-1-question_17.mp3',
          'scenario-1-question_18.mp3',
          'scenario-1-question_19.mp3',
          'scenario-1-question_20.mp3',
          'scenario-1-question_21.mp3',
          'scenario-1-question_22.mp3',
          'scenario-1-question_23.mp3',
          'scenario-1-question_24.mp3',
          'scenario-1-question_25.mp3',
          'scenario-1-question_26.mp3',
          'scenario-1-question_27.mp3',
          'scenario-1-question_28.mp3',
          'scenario-1-question_29.mp3',
          'scenario-1-question_30.mp3',
          'scenario-1-question_31.mp3',
          'scenario-1-question_32.mp3',
          'scenario-1-question_33.mp3',
          'scenario-1-question_34.mp3',
          'scenario-1-question_35.mp3',
          'scenario-1-question_36.mp3']
results = []
operator_vid = "chris"
tactica_vid = "matilda"
count = 0


creds = json.load(open(os.environ['AGENTCONFIG']))
stt_handler = SttPipeline(config=creds, use_local=True)
def play_audio(fpath):
    playsound(fpath)

parser = argparse.ArgumentParser(prog='Tactical Emulation',description='Emulates Tactica')
parser.add_argument('-t','--type', default="civil")
parser.add_argument('-c', '--civilian', help="path to file to scenario questions", default="docs/civilian_scenario_ground.json")      # option that takes a value
parser.add_argument('-m', '--military', help="path to file containing military scenario questions", default="docs/military_scenario_ground.json")      # option that takes a value
args = parser.parse_args()


def test_scripted_scenes_salim(scene_type="civil", fpath=None):
    network_manager.send_message(f"1.1")
    if scene_type == "civil":
        for i in tqdm(civilian_audio_list):
            fname = os.path.splitext(i)[0]
            udp_sign = fname.split("_")[1]
            fpath = os.path.join(args.audiopath, i) # user audio
            play_audio(fpath)
            pred = stt_handler.get_text(fpath)
            answer, udp_signal, is_scene = rag.handle_user_text(user_text=pred,voice_id=tactica_vid,scene_type=scene_type)
            if is_scene:
                udp_s = f"1.{udp_sign}"
                print(f"Sending udp signal: {udp_s}")
                network_manager.send_message(udp_s)
            network_manager.send_message("start")
            play_audio(answer)
            network_manager.send_message("stop")
            time.sleep(2)
    if scene_type == "military":
        pass

def test_scripted_scenes(scene_type="civil"):
    network_manager.send_message(f"1.1")
    df = json.load(open((args.civilian if scene_type == "civil" else args.military)))
    for j in tqdm(df):
        fpath = j["audio"]# user audio
        play_audio(fpath)
        answer, udp_signal, is_scene = rag.handle_user_text(user_text=j['question'],voice_id=tactica_vid,scene_type=scene_type)
        if is_scene:
            print(f"Sending udp signal: {udp_signal}")
            network_manager.send_message(udp_signal)
            time.sleep(0.1)
        network_manager.send_message("start")
        play_audio(answer)
        network_manager.send_message("stop")
        time.sleep(2)

test_scripted_scenes("military")
