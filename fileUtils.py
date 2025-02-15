"""
Name: fileUtils.py
Description: Module to handle files
Author: MT
"""
import os
import os
from pathlib import Path
import csv
from tqdm import tqdm
import json
import shutil
from ttsHandler import TtsHandler
import random
import time
from docx import Document
import pandas as pd

creds = json.load(open(os.environ['AGENTCONFIG']))
class FileUtils:
    def __init__(self):
        self.docs_dir = Path("docs")
        self.audio_dir = Path("audios")
        self.history_dir = Path("history")
        self.temp_audio_dir = self.audio_dir/"wav"
        self.user_audio_dir = self.audio_dir/"user"
        self.scenario_file_mil = self.docs_dir/"scenario_mil.json"
        self.scenario_file_civ = self.docs_dir/"scenario.json"
        self.history_file = self.history_dir/"history.json"
        self.backup_qa_file = self.docs_dir/"backup_qna.json"
        self.feedback_file = self.history_dir/"feedback.json"
        self.udp_mappings_file = self.docs_dir/"udp_mappings.json"
        self.docs_path_civ = self.docs_dir/"scenario_civ.docx"
        self.docs_path_mil = self.docs_dir/"scenario_mil.docx"
        self.csv_file_civ = self.docs_dir/"scenario_civ.csv"
        self.csv_file_mil = self.docs_dir/"scenario_mil.csv"
        self.tts_handler = TtsHandler(config=creds)
        self.main_df = None
        self.setup_dirs()

    def save_feedback(self, data):
        if os.path.exists(self.feedback_file):
            old = json.load(open(self.feedback_file))
            old.append(data)
            with open(self.feedback_file, 'w') as f:
                json.dump(old, f)
            f.close()
        else:
            with open(self.feedback_file, 'w') as f:
                json.dump([data], f)
            f.close()

    def setup_dirs(self):
        try:
            if not os.path.exists(self.history_dir):
                os.makedirs(self.history_dir)
            if not os.path.exists(self.audio_dir):
                os.makedirs(self.audio_dir)
            if not os.path.exists(self.temp_audio_dir):
                os.makedirs(self.temp_audio_dir)
            if not os.path.exists(self.user_audio_dir):
                os.makedirs(self.user_audio_dir)
        except Exception as e:
            print(f"Exception occurred during folder setup: {e}")

    def extract_table_from_docx(self, docx_file):
        print(f"[+] Extracting tabulated data from {docx_file} [+]")
        document = Document(docx_file)
        with open(self.csv_file, mode='w', newline='', encoding='utf-8') as file:
            csv_writer = csv.writer(file)
            for table in document.tables:
                for row in table.rows:
                    csv_writer.writerow([cell.text.strip() for cell in row.cells])
                csv_writer.writerow([])
        print(f"[+] Generated csv dataset generated {self.csv_file} [+]")

    def handle_user_audio(self, text, voice_id: str):
        rand_uid = f"{1}_{random.randint(1,20000)}"
        fname = f"{rand_uid}.mp3"
        voice_name = "chris" if voice_id not in self.tts_handler.voice_ids.keys() else voice_id
        voice_dir = os.path.join(self.audio_dir, voice_name)
        if not os.path.exists(voice_dir):
            os.makedirs(voice_dir)
        speech_file_path = os.path.join(voice_dir, fname)
        if self.tts_handler.eleven_tts(text, speech_file_path, voice_name):
            return speech_file_path
        
    def get_fpath(self):
        rand_uid = f"user_{random.randint(1,200000)}"
        fname = f"{rand_uid}.wav"
        fpath = os.path.join(self.user_audio_dir, fname)
        return fpath
    
    def get_alt_fpath(self):
        rand_uid = f"user_{random.randint(1,200000)}"
        fname = f"{rand_uid}.ogg"
        fpath = os.path.join(self.user_audio_dir, fname)
        return fpath

    
    def refactor_csv_file(self, scene_type):
        print(f"[+] Refactoring the csv file [+]")
        csv_file = self.csv_file_civ if scene_type == "civil" else self.csv_file_mil
        main_df = pd.read_csv(csv_file)
        if "Operator" in main_df.columns:
            main_df.rename(columns={'Operator':'question','AI Commentry':'answer', 'Step':'udp'}, inplace=True)
            main_df.to_csv(self.csv_file, index=False)
        self.main_df = main_df

    def load_docs(self, scene_type="civil", voice_id=None):
        doc_path = self.docs_path_civ if scene_type == "civil" else self.docs_path_mil
        print(f"[+] Loading docx {doc_path} [+]")
        # self.extract_table_from_docx(doc_path)
        self.refactor_csv_file(scene_type)
        voice_name = "matilda" if voice_id not in self.tts_handler.voice_ids.keys() else voice_id
        main_dir = os.path.join(self.docs_dir, scene_type)
        if not os.path.exists(main_dir):
            os.makedirs(main_dir)
        voice_docs_dir = os.path.join(self.docs_dir, voice_name) #audios/civil/chris ,#active/c
        if not os.path.exists(voice_docs_dir):
            os.makedirs(voice_docs_dir)
        self.scenario_file = os.path.join(voice_docs_dir, f"scenario_{scene_type}.json")
        if not os.path.exists(self.scenario_file):
            q_a = []
            print(f"[+] Processing refactored dataframe[+]")
            for i,k in tqdm(self.main_df.iterrows()):
                pred = "1" if scene_type == "civil" else "2"
                q_a.append({
                    "udp":f"{pred}.{k['udp']}",
                    "question":k['question'],
                    "answer":str(self.handle_user_audio(k['answer'], voice_name))})
            with open(self.scenario_file, 'w') as f:
                json.dump(q_a, f)
            f.close()

# mil_background = []
# df = pd.read_csv("docs/scenario_civ.csv")
# output_folder = "sceneground/audioscivilian"
# fileUtils = FileUtils()


# for i, j in tqdm(df.iterrows()):
#     udp_sign = j['udp']
#     qtn = j['question']
#     audio_file = fileUtils.handle_user_audio(qtn, voice_id="chris")
#     nfpath = os.path.join(output_folder,f"scenario_1_{udp_sign}.mp3")
#     shutil.copy(audio_file, nfpath)
#     os.remove(audio_file)
#     mil_background.append({
#         "question":qtn,
#         "audio":nfpath
#     })
#     time.sleep(2)

# with open("docs/civilian_scenario_ground.json", 'w') as f:
#     json.dump(mil_background, f)
# f.close()