import os
import time
import tempfile
import json

import pandas as pd
import chardet
from io import StringIO
from docx import Document

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

from utils import get_available_openai_models
from streaming import StreamHandler

os.environ["OPENAI_API_KEY"] = "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"
DB_DIRECTORY = "./chroma"

TEMPLATE = (
    "Question: {question}\n\n"
    "Use the following pieces of context to answer the question.\n"
    "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n"
    "----------------\n"
    "{context}"
)


@st.cache_data
def chat_model_list():
    return get_available_openai_models(put_first='gpt-3.5-turbo', filter_by='gpt')


@st.cache_data
def embedding_model_list():
    return get_available_openai_models(filter_by='embedding')


def process_uploaded_file(uploaded_file):
    try:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        return file_path
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")


def get_saved_databases():
    dbs = []
    for item in os.listdir(DB_DIRECTORY):
        item_path = os.path.join(DB_DIRECTORY, item)
        if os.path.isdir(item_path):
            dbs.append(item)
    return dbs


class StreamlitChatView:
    def __init__(self, default_embeddings_model="text-embedding-ada-002"):
        st.set_page_config(page_title="RAG ChatGPT", page_icon="📚", layout="wide")
        with st.sidebar:
            st.title("RAG ChatGPT")
            with st.expander("Model parameters"):
                self.model_name = st.selectbox("Model:", options=chat_model_list())
                self.temperature = st.slider("Temperature", min_value=0., max_value=2., value=0.7, step=0.01)
                self.top_p = st.slider("Top p", min_value=0., max_value=1., value=1., step=0.01)
                self.frequency_penalty = st.slider("Frequency penalty", min_value=0., max_value=2., value=0., step=0.01)
                self.presence_penalty = st.slider("Presence penalty", min_value=0., max_value=2., value=0., step=0.01)
            with st.expander("Embeddings parameters"):
                self.embeddings_model_name = st.selectbox("Embeddings model:", options=embedding_model_list(),
                                                          index=embedding_model_list().index(default_embeddings_model))
            with st.expander("Prompts"):
                self.context_prompt = st.text_area("Context prompt", value=TEMPLATE)
            self.input_file = st.file_uploader("Upload File", type=["pdf", "docx", "doc", "json", ".xlsx"])
            self.selected_db = st.selectbox("Select saved database:", [""] + get_saved_databases(), index=0)
            self.csv_file = st.file_uploader("Upload csv for questions", type="csv")

        self.user_query = st.chat_input(placeholder="Ask me anything!")

    def add_message(self, message, author):
        with st.chat_message(author):
            st.markdown(message)

    def add_message_stream(self, author: str):
        assert author in ["user", "assistant"]
        return StreamHandler(st.chat_message(author).empty())


def read_csv(view):
    if view.csv_file:
        file_content = view.csv_file.read()
        result = chardet.detect(file_content)
        encoding = result['encoding']
        decoded_content = file_content.decode(encoding)
        file_buffer = StringIO(decoded_content)
        df = pd.read_csv(file_buffer)
        return df


def clear_chat(memory):
    memory.chat_memory.messages = []


def prepare_docs(view):
    file_path = process_uploaded_file(view.input_file)
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.pdf':
        loader = PyPDFLoader(file_path)
        data = loader.load()
        return data
    elif file_extension.lower() in ['.docx', '.doc']:
        document = Document(file_path)
        data = []
        for paragraph in document.paragraphs:
            data.append(paragraph.text)
        docs = []
        for text in data:
            document = Document()
            document.page_content = text
            document.metadata = {}
            docs.append(document)
        return docs
    elif file_extension.lower() == '.json':
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        docs = []
        for text in data:
            document = Document()
            document.page_content = text
            document.metadata = {}
            docs.append(document)
        return docs
    elif file_extension.lower() == '.xlsx':
        data = pd.read_excel(file_path)
        docs = []
        for text in data:
            document = Document()
            document.page_content = text
            document.metadata = {}
            docs.append(document)
        return docs
    else:
        raise ValueError("Unsupported file type")


def create_conversational_chain(view, memory, prompt, df, db):
    llm = ChatOpenAI(
        model_name=view.model_name,
        temperature=view.temperature,
        top_p=view.top_p,
        frequency_penalty=view.frequency_penalty,
        presence_penalty=view.presence_penalty
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        verbose=True,
        chain_type="stuff",
        get_chat_history=lambda h: h,
        combine_docs_chain_kwargs={'prompt': prompt},
        memory=memory,
    )

    for message in memory.chat_memory.messages:
        view.add_message(message.content, 'assistant' if message.type == 'ai' else 'user')

    if view.user_query:
        view.add_message(view.user_query, "user")
        response = conversation_chain({"question": view.user_query})
        view.add_message(response['answer'], "assistant")

    elif view.csv_file:
        for index, row in df.iterrows():
            value = df.iloc[index, 0]
            view.add_message(value, "user")
            response = conversation_chain({"question": value})
            view.add_message(response['answer'], "assistant")


# Initialize the StreamlitChatView
view = StreamlitChatView()

prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=view.context_prompt
)

msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, output_key='answer', return_messages=True)

df = read_csv(view)

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Chat with Docs using OpenAI")

with col2:
    if st.button("Clear ↺"):
        clear_chat(memory)


if view.input_file:
    data = prepare_docs(view)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)
    if os.path.exists(os.path.join(DB_DIRECTORY, view.input_file.name)):
        st.warning('already have a db for this', icon="⚠️")
    else:
        embeddings = OpenAIEmbeddings(model=view.embeddings_model_name)
        vectorstore = Chroma.from_documents(docs, embeddings)
        if st.sidebar.button("save to database"):
            # vectorstore.persist(persist_directory=os.path.join(DB_DIRECTORY, view.input_file.name))
            vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=os.path.join(DB_DIRECTORY, view.input_file.name))
            st.success('database saved successfully select from below', icon="✅")
        create_conversational_chain(view, memory, prompt, df, vectorstore)

elif view.selected_db:
    embeddings = OpenAIEmbeddings(model=view.embeddings_model_name)
    selected_db_path = os.path.join(DB_DIRECTORY, view.selected_db)
    vectorstore = Chroma(persist_directory=selected_db_path, embedding_function=embeddings)
    create_conversational_chain(view, memory, prompt, df, vectorstore)
elif view.selected_db and view.input_file:
    st.warning("select only one source( Either upload or select db )", icon="⚠️")
else:
    st.info('select a database or upload file to create db', icon="ℹ️")

