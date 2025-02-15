"""
Name: coldBoot.py
Description: module to process the scene docx file to generate the scenario json file
with audios attached to it.
Author: MT
"""
from fileUtils import FileUtils
from tqdm import tqdm
import time
import sys


fileUtils = FileUtils()
voice_ids = ["matilda"]
scenarios = ["civil", "military"]
for scene_type in scenarios:
    for i in tqdm(voice_ids):
        msg = f"Hello, I am Tactica !"
        fileUtils.handle_user_audio(msg, i)
        fileUtils.load_docs(scene_type, voice_id=i)
        time.sleep(1)
