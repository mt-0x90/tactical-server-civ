from rag import RagPipeline
from sttHandler import SttPipeline
import pandas as pd
from tqdm import tqdm
import sys
import time
import random
from network import NetworkManager
from teacher import Teacher
from playsound import playsound
import json
import os

creds = json.load(open(os.environ['AGENTCONFIG']))
stt_handler = SttPipeline(config=creds, has_cpp=True)

# if len(sys.argv) < 2:
#     print(f"Usage: python {sys.argv[0]} <audio-ground-path> <qa-path>")
#     sys.exit(1)

audio_dir = sys.argv[1]
# qa_fpath = sys.argv[2]

# qa_data = json.load(open(qa_fpath))
qa_data = []
# audio_dir = ""

def get_ground_text(udp):
    ground_text = next((item for item in qa_data if item.get('udp') == str(udp)), None)
    if ground_text:
        return ground_text['question']
    return ground_text

rag = RagPipeline()
network_manager = NetworkManager()


def test_rag_qa(user_query):
    answer = rag.handle_user_text(user_query)
    return answer

def play_audio(fpath):
    playsound(fpath)

qa_eval = []
audio_list = ['scenario-1-question_2.mp3',
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
voice_ids = ["roger", "aria", "eric", "charlie", "chris", "bill", "jessica"]
count = 0

# def test_scripted_scenes():
#     for i in tqdm(audio_list):
#         fname = os.path.splitext(i)[0]
#         udp_sign = fname.split("_")[1]
#         fpath = os.path.join(audio_dir, i) # user audio
#         play_audio(fpath)
#         pred = stt_handler.get_text(fpath)
#         # voice_id = random.choice(voice_ids)
#         voice_id = "jessica"
#         answer, udp_signal, is_scene = rag.handle_user_text(user_text=pred,voice_id=voice_id)
#         if is_scene:
#             print(f"Sending udp signal: {udp_signal}")
#             # network_manager.send_http(f"1.{str(udp_signal)}")
#         # network_manager.send_http("start")
#         play_audio(answer)
#         # network_manager.send_http("stop")
#         time.sleep(2)

teacher = Teacher()
def test_freeform(count):
    global results
    bot_qtn = teacher.run_chatbot("Ask me a question about UAE, Middle East and the world in general")
    print(f"Question: {bot_qtn}")
    current = time.time()
    query = rag.run_chatbot(bot_qtn)
    elapsed = time.time() -  current
    print(f"Falcon: {query}")
    verdict = teacher.run_chatbot(f"Is the answer, {query} correct or wrong for the question: {bot_qtn}")
    print(f"Bot: {verdict}")
    if count == 2:
        bot_qtn = "Who is the president of UAE?"
        print(f"Question: {bot_qtn}")
        current = time.time()
        query = rag.run_chatbot(bot_qtn)
        elapsed = time.time() -  current
        print(f"Falcon: {query}")
        verdict = teacher.run_chatbot(f"Is the answer, {query} correct or wrong for the question: {bot_qtn}")
        print(f"Bot: {verdict}")
        if verdict.lower() == "wrong." or verdict.lower() == "wrong":
            answer = teacher.run_chatbot(bot_qtn)
        else:
            answer = query
        results.append({
            "question":bot_qtn,
            "falcon":query,
            "answer":answer,
            "verdict":verdict,
            "time":elapsed})
    if count == 3:
        bot_qtn = "Who is the minister of AI in UAE?"
        print(f"Question: {bot_qtn}")
        current = time.time()
        query = rag.run_chatbot(bot_qtn)
        elapsed = time.time() -  current
        print(f"Falcon: {query}")
        verdict = teacher.run_chatbot(f"Is the answer, {query} correct or wrong for the question: {bot_qtn}")
        print(f"Bot: {verdict}")
        if verdict.lower() == "wrong." or verdict.lower() == "wrong":
            answer = teacher.run_chatbot(bot_qtn)
        else:
            answer = query
        results.append({
            "question":bot_qtn,
            "falcon":query,
            "answer":answer,
            "verdict":verdict,
            "time":elapsed})
    if verdict.lower() == "wrong." or verdict.lower() == "wrong":
        answer = teacher.run_chatbot(bot_qtn)
    else:
        answer = query
    results.append({
        "question":bot_qtn,
        "falcon":query,
        "answer":answer,
        "verdict":verdict,
        "time":elapsed
    })


while True:
    test_freeform(count)
    count += 1
    if count == 20:
        break
    if count % 2 == 0:
        pd.DataFrame(results).to_csv("qna_general.csv", index=False)
    time.sleep(1)

pd.DataFrame(results).to_csv("qna_general.csv", index=False)



# for i in tqdm(all_qa):
#     start = time.time()
#     pred = test_rag_qa(i['user_query'])
#     elapsed = time.time() - start
#     qa_eval.append({
#         "user_query":i['user_query'],
#         "script_answer":i['script_answer'],
#         "model_answer":pred,
#         "time":str(elapsed)
#     })
# results_df = pd.DataFrame(qa_eval)
# results_df.to_csv("results_qa_general.csv", index=False)
# test_scripted_scenes()
# while True:
#     user_input = input(": ")
#     if user_input.lower() == "q":
#         break
#     answer, udp_signal, is_scene = rag.handle_user_text(user_text=user_input)
#     play_audio(answer)
    