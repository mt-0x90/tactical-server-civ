"""
name:llm.py
description: llm module
name: MT
"""
import os
import torch
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaLLM
from langgraph.graph import START, StateGraph
import torch
from langchain_community.utilities import GoogleSerperAPIWrapper, DuckDuckGoSearchAPIWrapper
import random
import tqdm
from langchain.agents import initialize_agent, Tool
from langchain_core.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from openai import OpenAI
from langchain import hub
from ttsHandler import TtsHandler
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer, util
from fileUtils import FileUtils
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from network import NetworkManager
import json
import sys
import pandas as pd

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Use this tool to search the web for information."

    def _run(self, query: str):
        # Perform a search using an API (e.g., SerpAPI or a custom search service)
        google_serper = GoogleSerperAPIWrapper()
        google_serper_results = google_serper.run(query)
        return google_serper_results

    async def _arun(self, query: str):
        raise NotImplementedError("Async support not implemented yet.")

creds = json.load(open(os.environ['AGENTCONFIG']))
os.environ["TAVILY_API_KEY"] = creds['tavily']
os.environ["SERPER_API_KEY"] = creds['serper']
os.environ['OPENAI_API_KEY'] = json.load(open(os.environ['AGENTCONFIG']))['OPENAI_API_KEY']

class RagPipeline:
    def __init__(self, use_ollama=True):
        self.use_ollama = use_ollama
        self.fileUtils = FileUtils()
        self.tts_handler = TtsHandler(creds)
        self.vector_store = None
        self.main_df = None
        self.google_serper = GoogleSerperAPIWrapper()
        self.websearch_tool = WebSearchTool()
        self.network_manager = NetworkManager()
        self.web_search_results = {}
        self.duckduckgo = DuckDuckGoSearchAPIWrapper()
        self.search_prompt = PromptTemplate(
            input_variables=["query", "search_results"],
            template=(
                "You are a usefull assistant. The user provided a question. Using the search results as context, "
                "Search Results:\n{search_results}\n\n"
                "Return the correct answer. Be precise\n"
            ),
        )
        self.scenario_file = self.fileUtils.scenario_file_civ
        self.history_file = self.fileUtils.history_file
        self.model_sentence = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.text_model_id = "falcon3:latest" # llama3.1:latest, falcon3:3b (opps),
        self.torch_dtype = torch.float16
        self.device = "cuda" if torch.cuda.is_available() else "mps"
        self.template = "You are a useful assistant, your role is to answer user questions to your fullest ability. your answers should be precise and accurate. Limit your answers to maximum 2 sentences. Answer with only facts nothing else."
        self.human_template = "{question}"
        self.template_class = """
        You are a userful assistant, given the following categories and examples of input fitting that category to use as context:
        Category A = Maritime Security, Examples: Zoom into Khalifa Port Economic Zone, For the ships currently here, What is their  vessel movement for the last 24 hours?
        Category B = Alerts and Full Analytics, Examples: What is the latest known information about this ship?, What is the trajectory of this ship?
        Category C = Port Activities, and Hybrid Analytics, Examples: Zoom into the port itself, Enhance the optical imagery and detect all ships
        Category D = Urban Planning, Examples: What are the newly constructed buildings in Khalifa City in 2024?, How many vehicles are in the imagery in this area?
        Category E = Airports, Examples: Zoom into Zayed airport?, Zoom into the one connected to the terminal.
        Category F = System Information, Examples: What kind of data can the system support?, Open the notification tab
        Category G = General, Examples: Hello, Tactica, Zoom into the UAE, What are the commercial ports in the UAE?
        Category H = Chitchat, Examples: Who is president of UAE?, What is the currency used in UAE?, Why should I live in Dubai?

        Use as context to classify user input according to closest category. If you do not know classify it as other. Answer only with the category nothing else. Be precise.    
        """
        self.human_template_class = "{user_input}"
        self.template_guard = """
        You are a useful assistant, your role is to answer user questions to your fullest ability. your answers should be precise and accurate. Limit your answers to maximum 2 sentences. If the user's questions is related to any of following categories:
        Category A = Maritime Security, Examples: Zoom into Khalifa Port Economic Zone, For the ships currently here, What is their  vessel movement for the last 24 hours?
        Category B = Alerts and Full Analytics, Examples: What is the latest known information about this ship?, What is the trajectory of this ship?
        Category C = Port Activities, and Hybrid Analytics, Examples: Zoom into the port itself, Enhance the optical imagery and detect all ships
        Category D = Urban Planning, Examples: What are the newly constructed buildings in Khalifa City in 2024?, How many vehicles are in the imagery in this area?
        Category E = Airports, Examples: Zoom into Zayed airport?, Zoom into the one connected to the terminal.
        Category F = System Information, Examples: What kind of data can the system support?, Open the notification tab
        Category G = General, Examples: Zoom into the UAE, What are the commercial ports in the UAE?
        Category H = Stage Setup, Examples: Tactica, generate and zoom-in into an area of interest 900 km West to East by 350km North to South, centered over the Straits of Hormuz,Identify all Naval ports, military, and dual-use airfields within the area of interest,
        Identify all island-based fixed coastal defensive and anti-shipping installations,Identify and generate all fixed and mobile surface to air missile systems and aerial surveillance sites.
        Category I = Air Activity Intelligence & Analysis, Examples: Tactica, Which airfield has the most military activity?, Zoom into the airport and tell me What satellite imagery do you have over this airport?,Using the most recent high-resolution optical image, identify all military aircraft types at this airport.,
        Is it operational?,Why is there no optical imagery from the last week?,Analyze the latest two SAR imagery,Generate a 3D representation of F4 Phantom., What about Bandar Abbas International Airport? Identify all military airplanes there.,
        Can Bandar Abbas operate and house Iran's recently purchased SU-35 Flanker aircraft?, Can you elaborate on the Shelters part?, What are the specifications of the SU-35 Flanker?, Is it a threat to UAE military forces if deployed from Bandar Abbas or Konarak Airports?,
        Where was the latest location where the SU-35 has been detected within the area of interest?, To what rank, 
        Category J = NPP Alert, Examples: Yes. Give more information about the NPP Alert., I don't know much about this location. What is BNPP?, Show me more information about all the latest changes?, Generate and open the report on the Power Plant,
        Close it, label it as restricted, share it with my supervisor, and archive it. Also, increase thermal coverage of the area, Take me to Bandar Abbas Naval port and show me the imagery there, Tell me more about this port?,Using this imagery, identify all naval ships within the naval port of Bandar Abbas,
        Using the last optical imagery,  what changes are occurring within the naval port? , Highlight the changes using the latest SAR with listing them and if there is any abnormal activity? , I am interested in the IRGC area. What military equipment is present in the IRGC area?,
        Is there any operational activity happening here?, What are your recommendations for the departed ships?,Identify all military ships in the last optical imagery in Area 1 and label,Why are these ships there?, Zoom into the Bagheri drone ship and provide more information    
        Answer with I do not know the answer or say you do not have the updated data. Avoid providing long answers. limit your answer to just one sentence or at most 3 sentences. Do not say you are AI.
        If you did not understand the question, say you don't understand it.
        """
        self.human_template_guard = "{question}"
        self.audio_fpath = None
        self.audio_model = None
        self.text_model = None
        self.audio_processor = None
        self.audio_pipeline = None
        self.instantiate_models()
        #self.init_vector_store()
        self.chatbot_chain = self.get_chain(self.template, self.human_template)
        self.chatbot_chain_guard = self.get_chain(self.template_guard, self.human_template_guard)
        self.classifier_chain = self.get_chain(self.template_class, self.human_template_class)
        self.tool_call_chain = self.search_prompt | self.text_model
        #self.network_manager.send_message("1.2")
        

    def get_chain(self, template, h_template):
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system",template),
            ("user",h_template)
        ])
        chain = chat_prompt | self.text_model
        return chain
    
    def search_online(self, query):
        """Search the internet using DuckDuckGo and Google Serper."""
        google_serper_results = self.duckduckgo.run(query)
        tools = [self.websearch_tool]
        agent = initialize_agent(tools=tools, llm=self.text_model, agent="zero-shot-react-description")
        web_results = agent.invoke(f"Search for {query}")
        # Combine results from both tools
        combined_results = (
            f"Google Serper Results:\n{google_serper_results}\n\nOther web results:\n{web_results}"
        )
        return combined_results
    
    def extract_answer_from_search(self, query):
        search_results = self.search_online(query)
        raw_results = self.tool_call_chain.invoke({"web_results":search_results, "query":query})
        results = raw_results if self.use_ollama else raw_results.content
        return results


    def instantiate_models(self):
        if self.use_ollama:
            self.text_model = OllamaLLM(model=self.text_model_id)
        else:
            os.environ['OPENAI_API_KEY'] = json.load(open(os.environ['AGENTCONFIG']))['OPENAI_API_KEY']
            self.text_model = ChatOpenAI(model="gpt-4o")
        self.fileUtils.load_docs()
    
    def init_vector_store(self):
        embeddings = OpenAIEmbeddings()
        texts = [f"Question: {row['question']} Answer: {row['answer']}" for _, row in self.fileUtils.main_df.iterrows()]
        metadatas = [{"question": row["question"], "answer": row["answer"]} for _, row in self.fileUtils.main_df.iterrows()]
        self.vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    def create_retriever(self):
        embeddings = OpenAIEmbeddings()
        texts = self.fileUtils.main_df['question'].tolist()
        metadatas = [{"answer": answer} for answer in self.fileUtils.main_df['answer']]
        self.vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    def run_chatbot(self, user_query):
        model_output = self.chatbot_chain.invoke({"question":user_query})
        results = model_output if self.use_ollama else model_output.content
        return results
    
    def run_chatbot_with_guard(self, user_query):
        model_output = self.chatbot_chain_guard.invoke({"question":user_query})
        results = model_output if self.use_ollama else model_output.content
        return results

    def run_classification(self, user_input):
        model_output = self.classifier_chain.invoke({"user_input":user_input})
        return model_output

    def rag_pipeline_old(self, user_query):
        results = self.vector_store.similarity_search_with_score(user_query, k=1)
        THRESHOLD = 0.7
        if results and results[0][1] > THRESHOLD:
            # If a close match is found, return the answer
            matched_doc = results[0][0]
            return matched_doc.metadata['answer']
        else:
            # Otherwise, use the LLM to generate an answer
            return self.run_chatbot(user_query)
    
    def get_sim_score(self, answer, pred):
        embedding_1= self.model_sentence.encode(answer, convert_to_tensor=True)
        embedding_2 = self.model_sentence.encode(pred, convert_to_tensor=True)
        sim_score = round(util.pytorch_cos_sim(embedding_1, embedding_2)[0].detach().item(),3)
        return sim_score
    
    def check_history(self, user_text):
        with open(self.fileUtils.backup_qa_file) as f:
            data = json.load(f)
        answer = next((item for item in data if self.get_sim_score(item.get('question'), user_text) > 0.7), None)
        return answer
    
    def check_scenario(self, user_text, voice_id, scene_type):
        voice_name = "chris" if voice_id not in self.tts_handler.voice_ids.keys() else voice_id
        scenario_dir = os.path.join(self.fileUtils.docs_dir, voice_name)
        self.scenario_file = os.path.join(scenario_dir, f"scenario_{scene_type}.json")
        with open(self.scenario_file) as f:
            data = json.load(f)
        answer = next((item for item in data if self.get_sim_score(item.get('question'), user_text) > 0.7), None)
        return answer
    
    def get_udp_signal(self, user_text):
        with open(self.fileUtils.udp_mappings_file) as f:
            data = json.load(f)
        answer = next((item for item in data if self.get_sim_score(item.get('question'), user_text) > 0.8), None)
        return answer
    
    # Define application steps
    def retrieve(self, state: State):
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}
    
    def generate(self, state: State):
        prompt = hub.pull("rlm/rag-prompt")
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        # print(docs_content)
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.text_model.invoke(messages)
        return {"answer": response}

    def rag_pipeline(self, user_query):
        graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()
        response = graph.invoke({"question": f"Given the question: {user_query}"})
        return response['answer']
    
    def handle_user_audio(self, text, voice_id):
        rand_uid = f"{1}_{random.randint(1,20000)}"
        fname = f"{rand_uid}.mp3"
        voice_name = "matilda"
        voice_dir = os.path.join(self.fileUtils.audio_dir, voice_name)
        if not os.path.exists(voice_dir):
            os.makedirs(voice_dir)
        speech_file_path = os.path.join(voice_dir, fname)
        response = self.tts_handler.eleven_tts(text, speech_file_path, voice_id)
        return speech_file_path
    
    
    def handle_user_text(self, user_text, voice_id="chris", is_local=False, scene_type="civil"):
        scen_t = "civil" if scene_type == "civilian" else scene_type
        his_answer = self.check_scenario(user_text, voice_id.lower(), scen_t)
        if his_answer:
            udp_signal = his_answer['udp']
            return his_answer['answer'], udp_signal, True
        known_qa = self.check_history(user_text)
        if known_qa:
            bot_answer = known_qa['answer']
        else:
            bot_answer = self.run_chatbot_with_guard(user_text).replace("Falcon","")
        if is_local:
            return bot_answer, None, False
        return self.handle_user_audio(bot_answer, voice_id.lower()), None, False

# import time
# rag = RagPipeline()
# df = json.load(open(sys.argv[1]))
# for i in tqdm.tqdm(df):
#     print(f"Question: {i['question']}")
#     answer, _, _ = rag.handle_user_text(i['question'], is_local=True)
#     print(f"Bot: {answer}")
#     time.sleep(1)
# results = []
# for i,j in df.iterrows():
#     result = rag.run_classification(j['question'])
#     print(f"Text: {j['question']} :: Category: {result}")
#     results.append({
#         'question':j['question'],
#         'category':result
#     })

# pd.DataFrame(results).to_csv("rag_test.csv", index=False)
# while True:
#     user_input = input(":> ")
#     if user_input == "q":
#         break
#     answer = rag.run_chatbot(user_input)
#     print(f"Bot: {answer}")
