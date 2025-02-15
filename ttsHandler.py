import os
from openai import OpenAI
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import requests


class TtsHandler:
    def __init__(self, config, use_openai=False, use_kokoro=False):
        self.use_openai = use_openai
        self.use_kokoro = use_kokoro
        self.voice_ids = {
            "charlie":"IKne3meq5aSn9XLyUdCD",
            "eric":"cjVigY5qzO86Huf0OWal",
            "chris":"iP95p4xoKVk53GoZ742B",
            "jessica":"cgSgspJ2msm6clMCkdW9",
            "matilda":"XrExE9yKIg1WjnnlVkGX",
            "mt":"tBEGppNdv1PGraDSEUp0",
            "zara":"l8UYco4ZYEkkkmX0nR7g"
        }
        self.config = config
        self.audio_llm = None
        self.setup_llms()


    def setup_llms(self):
        if self.use_openai:
            os.environ["OPENAI_API_KEY"] = self.config['OPENAI_API_KEY']
            self.audio_llm = OpenAI()
            return False
        if self.use_kokoro:
            """
            !git lfs install
            !git clone https://huggingface.co/hexgrad/Kokoro-82M
            %cd Kokoro-82M
            !apt-get -qq -y install espeak-ng > /dev/null 2>&1
            brew install espeak-ng
            !pip install -q phonemizer torch transformers scipy munch
            """
            print(f"Kokoro not implemented yet")
            return False
        self.audio_llm = ElevenLabs(api_key=self.config['ELEVENLABS_API_KEY'])
        return True
    
    def openai_tts(self, text: str, fpath) -> bool:
        response = self.audio_llm.audio.speech.create(
            model="tts-1",
            voice="alloy",input=text)
        response.stream_to_file(fpath)
        return os.path.exists(fpath)
    
    def eleven_tts(self, text: str, fpath: str, voice_id: str) -> bool:
        #selected_voice_id = "eric" if voice_id not in self.voice_ids.keys() else voice_id
        selected_voice_id = voice_id
        selected_voice = self.voice_ids[selected_voice_id]
        response = self.audio_llm.text_to_speech.convert(
            voice_id=selected_voice,
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_flash_v2_5",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.7,
                style=0.0,
                use_speaker_boost=True,
                ),
            )
        with open(fpath, 'wb') as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)
        return os.path.exists(fpath)
        
