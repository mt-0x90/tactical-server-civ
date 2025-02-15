from sttHandler import SttPipeline
import os
import pandas as pd
import sys
import time
from tqdm import tqdm
import json

creds = json.load(open(os.environ['AGENTCONFIG']))
stt_handler = SttPipeline(config=creds, has_cpp=True)

if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <audio-ground-path> <qa-path>")
    sys.exit(1)

print(sys.argv)
audio_dir = sys.argv[1]
qa_fpath = sys.argv[2]

qa_data = json.load(open(qa_fpath))

def get_ground_text(udp):
    ground_text = next((item for item in qa_data if item.get('udp') == str(udp)), None)
    if ground_text:
        return ground_text['question']
    return ground_text

results = []
for i in tqdm(os.listdir(audio_dir)):
    fname = os.path.splitext(i)[0]
    udp_sign = fname.split("_")[1]
    fpath = os.path.join(audio_dir, i)
    current = time.time()
    pred = stt_handler.get_text(fpath)
    elapsed = time.time() - current
    results.append({
        "ground":get_ground_text(udp_sign),
        "prediction":pred,
        "time":elapsed,
        "method":"mac_m4"
    })

pd.DataFrame(results).to_csv("stt_test_mac_m4.csv", index=False)
