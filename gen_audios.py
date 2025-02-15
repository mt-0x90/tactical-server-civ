import pandas as pd
from tqdm import tqdm
import os
import json
from rag import RagPipeline
from ttsHandler import TtsHandler

creds = json.load(open(os.environ['AGENTCONFIG']))