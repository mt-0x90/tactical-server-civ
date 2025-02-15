import os
from flask import Flask, request, jsonify
import base64
import random
from pathlib import Path
from stt import SttPipeline
from rag import RagPipeline

app = Flask(__name__)
rag = RagPipeline(True)
stt = SttPipeline()

@app.route('/upload', methods=['POST'])
def upload_file():
    data = request.json
    file_name = data['fileName']
    file_content = data['fileContent']
    # Decode the Base64 content and save the file
    print(file_name)
    with open(file_name, "wb") as f:
        f.write(base64.b64decode(file_content))
    f.close()
    user_text = stt.get_text(file_name, file_name.split(".")[0])
    fpath, flag = rag.handle_user_text(user_text)
    if flag == 1:
        with open(fpath, "rb") as mp3_file:
            mp3_bytes = mp3_file.read()
        base64_string = base64.b64encode(mp3_bytes).decode('utf-8')
        return jsonify({"message": base64_string, "flag":"1"}), 200
    if flag == 0:
        return jsonify({"message": fpath, "flag":"0"}), 200

if __name__ == '__main__':
    app.run(debug=True)

