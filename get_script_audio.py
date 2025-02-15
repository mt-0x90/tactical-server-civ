import sys
import shutil
import os
from tqdm import tqdm
import json

src_dir = sys.argv[1]
dest_dir = sys.argv[2]


data = json.load(open(src_dir))
meta_data = []
for i in tqdm(data):
    i_path = i['answer']
    udp = i['udp']
    fpath = f"scenario_{udp}.mp3"
    d_path = os.path.join(dest_dir, fpath)
    shutil.copy(i_path, d_path)
    meta_data.append({"udp":udp,"audio":d_path})

with open("meta_data.json", 'w') as f:
    json.dump(meta_data, f)
f.close()