"""
Name:sttHandler.py
Description: Speech to text module
Author: MT
"""
import os
import librosa
import subprocess
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, Wav2Vec2Processor, Wav2Vec2ForCTC, AutoTokenizer, AutoModel
import time
from deepgram import DeepgramClient, PrerecordedOptions
import wave
from vosk import Model, KaldiRecognizer, SetLogLevel
import json
import shutil
import pandas as pd
import time
from fileUtils import FileUtils
from executor import Executor
from pathlib import Path
from openai import OpenAI
import sys
from tqdm import tqdm


whispercpp = Path("whisper.cpp")
creds = json.load(open(os.environ['AGENTCONFIG']))
# os.environ["OPENAI_API_KEY"] = creds['OPENAI_API_KEY']

class SttPipeline:
	def __init__(self, config, has_cpp=False, use_hf=False, use_remote=False, use_local=False):
		self.model_id = "openai/whisper-large-v3-turbo"
		self.config = config
		self.has_cpp = has_cpp
		self.use_hf = use_hf
		self.use_remote = use_remote
		self.use_local = use_local
		self.executor = Executor()
		self.fileUtils = FileUtils()
		#self.whisper_model_path = str(whispercpp/"models/ggml-large-v3-turbo.bin")
		self.whisper_model_path = "/Users/mt/projects/whisper.cpp/models/ggml-large-v3-turbo.bin"
		#self.whisper_bin = str(whispercpp/"build/bin/whisper-cli")
		self.whisper_bin = "/Users/mt/projects/whisper.cpp/build/bin/whisper-cli"
		self.audio_model = None
		self.remote_whisper = OpenAI(api_key="EMPTY", base_url="https://qlq73bshz87qgp-8000.proxy.runpod.net/v1/")
		self.remote_model_id = "deepdml/faster-whisper-large-v3-turbo-ct2"
		self.audio_pipeline = None
		self.deepgram = DeepgramClient(creds['deepgram'])
		self.wave_netmodel = None
		self.torch_dtype = torch.float16
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		if self.use_hf:
			self.instantiate_models()

	def remote_stt_whisper(self, fpath, lang="en"):
		try:
			audio_stream = open(fpath, 'rb')
			transcript = self.remote_whisper.audio.transcriptions.create(
				model=self.remote_model_id,
				file=audio_stream,
				language=lang,
				response_format="text")
			return transcript
		except Exception as e:
			print(e)
			return None
        

	def instantiate_models(self):
		self.audio_model = AutoModelForSpeechSeq2Seq.from_pretrained(
			self.model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
		self.audio_model.to(self.device)
		audio_processor = AutoProcessor.from_pretrained(self.model_id)
		self.audio_pipeline = pipeline(
			"automatic-speech-recognition",
			model=self.audio_model,
			tokenizer=audio_processor.tokenizer, feature_extractor=audio_processor.feature_extractor, torch_dtype=self.torch_dtype,
			device=self.device)
		
	def execute_command(self, cmd):
		ret = subprocess.run(cmd, shell=True, capture_output=True, text=True)
		return ret
	def handle_deepgram_transcribe(self, fpath):
		with open(fpath, 'rb') as buffer_data:
			payload = { 'buffer': buffer_data }
			options = PrerecordedOptions(smart_format=True, model="nova-2", language="en-US")
			response = self.deepgram.listen.prerecorded.v('1').transcribe_file(payload, options)
			results = response['results']['channels'][0]['alternatives'][0]['transcript']
			return results
		
	def handle_voice_proc(self, fpath, fname):
		txt_file = f'{fname}.txt'
		txt_fpath = os.path.join(self.fileUtils.audio_dir, txt_file)
		command = f'whisper "{fpath}" --model turbo --language English --output_dir {self.fileUtils.audio_dir} --output_format txt'
		os.system(command)
		#self.executor.execute_command(command)
		if os.path.exists(txt_fpath):
			results = open(txt_fpath).read()
			return results, txt_fpath
		
	def transcribe_voice(self, fpath):
		print(f"Hellooo: {fpath}")
		command = f"{self.whisper_bin} -m {self.whisper_model_path} {str(fpath)} -otxt"
		#command = f"./whisper-cli.sh {fpath}"
		os.system(command)
		txt_file = f"{fpath}.txt"
		return open(txt_file).read()

	def handle_voice_trans(self, fpath):
		results = self.audio_pipeline(fpath)
		return results['text']
	
	def convert_to_wav(self, fpath, fname):
		n_fpath = f"{fname}.wav"
		try:
			nfpath = os.path.join(self.fileUtils.temp_audio_dir, n_fpath)
			if os.path.exists(nfpath):
				os.remove(nfpath)
			command = f"ffmpeg -i {fpath} -ar 16000 -ac 1 -c:a pcm_s16le {nfpath}"
			self.execute_command(command)
			if os.path.exists(nfpath):
				return nfpath
		except Exception as e:
			print(f"exception: {e}")
			print(f"Error during conversion: {e}")
			return str(e)

	def get_text(self, fpath, is_wav=False):
		if os.path.exists(fpath):
			fname = os.path.basename(fpath)
			if os.path.splitext(fpath)[1] == ".3gp":
				fpath = self.convert_to_wav(fpath, fname)
			if self.use_hf:
				current = time.time()
				results = self.handle_voice_trans(fpath)
				elapsed = time.time() - current
				print(f"Detected text: {results}")
				print(f"Transcribing took {elapsed} seconds")
				return results
			if self.use_remote:
				current = time.time()
				results = self.remote_stt_whisper(fpath)
				elapsed = time.time() - current
				print(f"Detected text: {results}")
				print(f"Transcribing took {elapsed} seconds")
				return results
			if self.has_cpp:
				current = time.time()
				print(f"Hello: {fpath}")
				#nfpath = self.convert_to_wav(fpath, fname) if not is_wav else fpath
				nfpath = self.convert_to_wav(fpath, fname)
				results = self.transcribe_voice(nfpath)
				elapsed = time.time() - current
				print(f"Detected text: {results}")
				print(f"Transcribing took {elapsed} seconds")
				return results
			if self.use_local:
				current = time.time()
				results, fpath = self.handle_voice_proc(fpath, os.path.splitext(fname)[0])
				# shutil.rmtree(fpath)
				elapsed = time.time() - current
				print(f"Detected text: {results}")
				print(f"Transcribing took {elapsed} seconds")
				return results
			

# model = Model(lang="en-us")

# def handle_with_vosk(fpath):
# 	try:
# 		wf = wave.open(fpath, 'rb')
# 		if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
# 			print(f"Audio file must be wav")
# 			sys.exit(1)
# 		rec = KaldiRecognizer(model, wf.getframerate())
# 		rec.SetWords(True)
# 		rec.SetPartialWords(True)
# 		results = ""
# 		while True:
# 			data = wf.readframes(4000)
# 			if len(data) == 0:
# 				break
# 			if rec.AcceptWaveform(data):
# 				results = rec.Result()
# 		return results
# 	except Exception as e:
# 		print(f"Exception: {e}")
# 		return None
		
# sst = SttPipeline(creds)
# audio_list = os.listdir(sys.argv[1])
# tran_res = []
# for i in tqdm(audio_list):
# 	fpath = os.path.join(sys.argv[1], i)
# 	nfpath = sst.convert_to_wav(fpath, "temp")
# 	current = time.time()
# 	results = handle_with_vosk(nfpath)
# 	elapsed = time.time() - current
# 	os.remove(nfpath)
# 	print(f"Text: {results}")
# 	print(f"Elapsed: {elapsed}")
# 	tran_res.append({
# 		"transcribe":results,
# 		"time":elapsed
#     })
	
# pd.DataFrame(tran_res).to_csv("stt_test_vosk.csv", index=False)