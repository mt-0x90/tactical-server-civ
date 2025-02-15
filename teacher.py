from langchain_ollama import OllamaLLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_openai import ChatOpenAI
import os
import json


class Teacher:
    def __init__(self, use_ollama=False):
        self.use_ollama = use_ollama
        self.template = "You are a a useful assistant, answer all questions the user asks. Provide just the answer, no need for any extra words. When asked if an answer to a question is correct you respond with correct if its correct or wrong if wrong."
        self.human_template = "{query}"
        self.text_model_id = "falcon3:latest"
        self.text_model = None
        self.workflow = StateGraph(state_schema=MessagesState)
        self.history = []
        self.instatiate_model()
        #self.chatbot_chain = self.get_chain(self.template, self.human_template)
        self.app = None
        self.setup_checkpointer()

    def instatiate_model(self):
        if self.use_ollama:
            self.text_model = OllamaLLM(model=self.text_model_id)
        else:
            os.environ['OPENAI_API_KEY'] = json.load(open(os.environ['AGENTCONFIG']))['OPENAI_API_KEY']
            self.text_model = ChatOpenAI(model="gpt-4o")

    def call_model(self, state: MessagesState):
        system_prompt = (
            "You are a helpful assistant. "
            "answer all questions the user asks. Provide just the answer, no need for any extra words. When asked if an answer to a question is correct you respond with correct if its correct or wrong if wrong.")
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = self.text_model.invoke(messages)
        return {"messages": response}
    
    def setup_checkpointer(self):
        self.workflow.add_node("model", self.call_model)
        self.workflow.add_edge(START, "model")
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)

    def get_chain(self, template, h_template):
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system",template),
            ("user",h_template)
        ])
        chain = chat_prompt | self.text_model
        return chain
    
    def run_chatbot(self, user_query):
        #model_output = self.chatbot_chain.invoke({"query":user_query})
        model_output = self.app.invoke(
            {"messages": [HumanMessage(content=f"{user_query}")]},
            config={"configurable": {"thread_id": "1"}},)
        if not self.use_ollama:
            results = model_output['messages'][-1].content
            return results
        #results = model_output if self.use_ollama else model_output.content
        return model_output['messages'][-1]
    

# teacher = Teacher()
# while True:
#     bot_qtn = teacher.run_chatbot("Ask me a question about UAE, Middle East and the world in general")
#     print(f"Question: {bot_qtn}")
#     answer = input(": ")
#     query = answer.strip()
#     if query.lower() == "q":
#         break
#     verdict = teacher.run_chatbot(f"Is the answer, {query} correct or wrong for the question: {bot_qtn}")
#     print(f"Bot: {verdict}")