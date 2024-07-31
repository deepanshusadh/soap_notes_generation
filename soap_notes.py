import os
import openai
from langchain_openai import ChatOpenAI
from datasets import load_dataset
import csv
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
import evaluate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough

#----------------------------------------------------------------------------

#load example conversation and soap note for one shot prompt
with open("call_conv_ex.txt", 'r') as text_file:
   conversation_x=text_file.read()

with open("soap_notes_ex.txt", 'r') as text_file:
   soap_x=text_file.read()

#load open source embedding 
embeddings = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en-v1.5", model_kwargs = {'device': 'cpu'})
# load FAISS vectorstore, which was saved in python notebook for test evaluation
vectorstore=FAISS.load_local("vectorstore", embeddings ,allow_dangerous_deserialization=True)

#load OpenAI model
os.environ["OPENAI_API_KEY"] = "Enter your Open AI key"
llm = ChatOpenAI(model="gpt-4")

#keeping k as 5, to limit last 5 chat history only
def filter_messages(messages, k=5):
    return messages[-k:]

#stores all the chat message history
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


#-------------------------RAG----------------------------------------------------------------

#prompt template for rag based generation
prompt_rag = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are expert in writing SOAP notes from the conversation.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

#make a chain
chain_rag = (
    RunnablePassthrough.assign(messages=lambda x: filter_messages(x["messages"]))
    | prompt_rag
    | llm
)

#final function of generation which includes generation chain and message history
with_message_history_rag = RunnableWithMessageHistory(chain_rag, get_session_history, input_messages_key="messages")


# history
def retrieve_response(conversation: str) -> str:
  """
  This function takes in conversation and return most similar conversation from the FAISS vectorstore
  """
  unique_docs = vectorstore.similarity_search(conversation)
  conv_soap_pair=f""" **Conversation:**\n{unique_docs[0].page_content}\n**SOAP Notes:**\n{unique_docs[0].metadata["soap_notes"]}"""

  return conv_soap_pair

def generate_response(conversation: str, conv_soap_pair : str,config_id : str) -> str: 
   
    """  This function generate soap notes taking a conversation, 
    conv_soap_pair as context for format and config_id for chat_history management.
    """

    response_rag = with_message_history_rag.invoke(
    {"messages": [HumanMessage(content=f"Write SOAP Notes from the provided conversation: \n {conversation}.Here is an example of a conversation and its respective SOAP notes {conv_soap_pair}. Use the same format of the soap notes as provided in the example")]
     },
     config={"configurable": {"session_id": config_id}}
    )
  
    return response_rag.content

def chat(prompt : str,conv_soap_pair : str ,config_id : str) -> str:
    """
    This function gives user a functionality to chat with the llm keeping chat history as context (only for rag based generation)
    """
    response_prompt = with_message_history_rag.invoke(
    {"messages": [HumanMessage(content=prompt)],"context":conv_soap_pair
     },
     config={"configurable": {"session_id": config_id}}
    )
  
    return response_prompt.content

#----------------------------Direct Prompting (One Shot)------------------------------------------------------------------------------------

#Prompt Template for direct prompting
prompt_direct = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are expert in writing SOAP notes from the conversation."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

#create a chain
chain_direct = (
    RunnablePassthrough.assign(messages=lambda x: filter_messages(x["messages"]))
    | prompt_direct
    | llm
)

# final generation function taking generation chain and managing chat history
with_message_history_direct = RunnableWithMessageHistory(chain_direct, get_session_history, input_messages_key="messages")

def one_shot_prompt(conversation : str) -> str:
  """
  This function takes in conversation and craft it in a one shot prompt, 
  which can be used for generating soap notes of the conversation.
  """
  prompt=f"""Write SOAP Notes from the provided conversation.
  <conversation>
  Here is the conversation from which you need to write SOAP notes:
  {conversation}
  </conversation>
  <example>
  Here is an example of conversation and its respective SOAP notes:
  Conversation: {conversation_x}
  SOAP Notes: {soap_x}
  </example>
  <format>
  Here is the format of the response
  Subjective (S): Patient's reported symptoms and medical history.
  Objective (O): Measurable and observable clinical data.
  Assessment (A): Professional interpretation and diagnosis.
  Plan (P): Strategy for treatment and management.
  </format>
"""
  return prompt

def generate_response_direct(conversation : str,config_id : str) -> str: 
   """
   This function takes conversation to craft it in one shot prompt and config id to manage chat history
   """
   response_direct= with_message_history_direct.invoke(
    {"messages": [HumanMessage(content=one_shot_prompt(conversation))]
     },
     config={"configurable": {"session_id": config_id}}
    )
   
   return response_direct.content

def chat_direct(prompt : str,config_id: str) -> str:
    """
    This function gives user a functionality to chat with the llm keeping chat history as context 
    (only for dirct prompting based generation)
    """
    response_prompt = with_message_history_direct.invoke(
    {"messages": [HumanMessage(content=prompt)]
     },
     config={"configurable": {"session_id": config_id}}
    )
  
    return response_prompt.content
