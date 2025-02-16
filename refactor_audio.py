import sys
import pandas as pd
from tqdm import tqdm
import os
import json
import random
from ttsHandler import TtsHandler
creds = json.load(open(os.environ['AGENTCONFIG']))

old_json = sys.argv[1]
new_csv = sys.argv[2]

tts = TtsHandler(config=creds)
data = json.load(open(old_json))
df = pd.read_csv(new_csv)

def generate_audio(fpath, text):
    if os.path.exists(fpath):
        os.remove(fpath)
    if tts.eleven_tts(text, fpath, "matilda"):
        return fpath
    
refactor_civil = ["1.2","1.3", "1.9","1.13","1.22","1.24","1.25","1.26","1.27","1.28", "1.29","1.30","1.31","1.32","1.33","1.34","1.35","1.36","1.37","1.38"]
#refactor_mil = ["2.2","2.7", "2.9","2.12","2.14","2.20","2.21","2.23","2.24","2.25","2.26","2.27","2.28", "2.29","2.30","2.31","2.32","2.33","2.34","2.35","2.36","2.37","2.38","2.39"]
refactor_mil = ["2.7", "2.8","2.11","2.14","2.15","2.16","2.17","2.22","2.24"]
# Build a mapping from id to dictionary for quick lookup
data_mapping = {item['udp']: item for item in data}
# Iterate over each row in the DataFrame
for _, row in tqdm(df.iterrows()):
    # Convert row to a dictionary
    row_data = row.to_dict()
    row_id = f"2.{row_data.get('udp')}"  # Ensure 'id' is the key used in both CSV and dictionaries
    if row_id in data_mapping:
        # If id exists, update the corresponding dictionary with new data
        # if row_id in refactor_mil:
        #     qtn = data_mapping[row_id]['question']
        #     fpath = data_mapping[row_id]['answer']
        #     udp = data_mapping[row_id]['udp']
        #     row_data['udp'] = row_id
        #     nfpath = generate_audio(fpath, row_data['answer'])
        #     row_data['answer'] = nfpath
        #     data_mapping[row_id].update(row_data)
        if row_id != "1.1" or row_id != "2.1":
            fpath = data_mapping[row_id]['answer']
            udp = data_mapping[row_id]['udp']
            row_data['udp'] = row_id
            row_data['answer'] = fpath
            data_mapping[row_id].update(row_data)
    else:
        # If id does not exist, append the new row data to the list
        # row_data['answer'] = generate_audio(fpath)
        data.append(row_data)
        nfpath = generate_audio(f"audios/matilda/{random.randint(1,2000)}.mp3", row_data['answer'])
        row_data['answer'] = nfpath
        data_mapping[row_id] = row_data
    # print(row_data)
    # break
new_data = []
for i in data_mapping.keys():
    new_data.append(data_mapping[i])
with open(old_json, 'w') as f:
    json.dump(new_data, f)
f.close()