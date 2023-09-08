import os
import time
import gradio as gr

from typing import Iterable

from langchain.document_loaders import OnlinePDFLoader, PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain import FAISS

from dotenv import load_dotenv

load_dotenv()

ENVIRONMENT = os.environ.get("ENVIRONMENT") or "development"

AZURE_OPENAI_API_TYPE = os.environ.get("OPENAI_API_TYPE") or "azure"
AZURE_OPENAI_SERVICE_BASE = os.environ.get("OPENAI_API_BASE") or "https://myopenai.openai.azure.com"
AZURE_OPENAI_SERVICE_API_KEY = os.environ.get("OPENAI_API_KEY")
AZURE_OPENAI_SERVICE_API_VERSION = os.environ.get("OPENAI_API_VERSION") or "2023-05-15"
AZURE_OPENAI_SERVICE_EMBEDDING_DEPLOYMENT = os.environ.get("OPENAI_API_EMBEDDING_DEPLOYMENT") or "text-embedding-ada-002"
AZURE_OPENAI_SERVICE_EMBEDDING_MODEL = os.environ.get("OPENAI_API_EMBEDDING_MODEL") or "text-embedding-ada-002"
AZURE_OPENAI_SERVICE_CHAT_DEPLOYMENT = os.environ.get("OPENAI_API_CHAT_DEPLOYMENT") or "chat"
AZURE_OPENAI_SERVICE_CHAT_MODEL = os.environ.get("OPENAI_API_CHAT_MODEL") or "chat"

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE") or 1000)
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP") or 200)

USERNAME_LOGIN = os.environ.get("USERNAME_LOGIN") or "admin"
PASSWORD_LOGIN = os.environ.get("PASSWORD_LOGIN") or "password"

def chunkify(arr: Iterable, size: int = 8):
    for i in range(0, len(arr), size):
        yield arr[i : i + size]

def loading_pdf():
    return "Loading..."


def load_pdf(pdf):
    loader = PyPDFLoader(pdf.name)
    return loader.load()

def split_document(method, documents):
    text_splitter = None
    if method == "CharacterTextSplitter":
        text_splitter = CharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    elif method == "RecursiveCharacterTextSplitter":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    else:
        # throw error
        raise Exception("No method provided")

    return text_splitter.split_documents(documents)

def init_faiss(chunks, embeddings):
    # Convert the chunks of text into embeddings to form a knowledge base
    vecotorstore = FAISS.from_texts([""], embeddings)
    for chunk in chunkify(chunks):
        vecotorstore.add_documents(chunk)

    return vecotorstore.as_retriever()

def init_chroma(chunks, embeddings):
    # Convert the chunks of text into embeddings to form a knowledge base
    vectorstore = Chroma(embedding_function=embeddings)
    for chunk in chunkify(chunks):
        vectorstore.add_documents(chunk)

    return vectorstore.as_retriever()

def pdf_changes(pdf_doc, search_engine, split_method):
    embeddings = OpenAIEmbeddings(
        model=AZURE_OPENAI_SERVICE_EMBEDDING_MODEL,
        deployment=AZURE_OPENAI_SERVICE_EMBEDDING_DEPLOYMENT,
    ) #type: ignore

    # load pdf
    pdf_doc = load_pdf(pdf_doc)

    # split pdf
    chunks = None
    if split_method == "CharacterTextSplitter":
        chunks = split_document("CharacterTextSplitter", pdf_doc)
    elif split_method == "RecursiveCharacterTextSplitter":
        chunks = split_document("RecursiveCharacterTextSplitter", pdf_doc)
    else:
        # throw error
        raise Exception("No method provided")

    # setup search engine
    vector_store = None
    if search_engine == "FAISS":
        vector_store = init_faiss(chunks, embeddings)
    elif search_engine == "Chroma":
        vector_store = init_chroma(chunks, embeddings)
    else:
        # throw error
        raise Exception("No method provided")

    global qa
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain
    qa = RetrievalQA.from_chain_type(
        llm=AzureOpenAI(
            model=AZURE_OPENAI_SERVICE_CHAT_MODEL,
            deployment_name=AZURE_OPENAI_SERVICE_CHAT_DEPLOYMENT,
            temperature=0.1,
            max_tokens=200,
        ),
        retriever=vector_store,
        return_source_documents=True,
        chain_type="refine",
    )

    return "Ready"

def add_text(history, text):
    history = history + [(text, None)]

    return history, ""

def bot(history):
    response = infer(history[-1][0], history)

    history[-1][1] = ""

    for character in response:
        history[-1][1] += character
        yield history

def infer(question, history):
    res = []
    for human, ai in history[:-1]:
        pair = (human, ai)
        res.append(pair)

    chat_history = res
    query = question
    result = qa({"query": query, "chat_history": chat_history})

    print(result)

    return result["result"].replace("<|im_end|>", "")

css="""
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
"""

title = """
<div style="text-align: center;max-width: 700px;">
    <h1>Chat with PDF using Azure OpenAI</h1>
    <p style="text-align: center;">Upload a .PDF from your computer, click the "Load PDF to LangChain" button, <br />
    when everything is ready, you can start asking questions about the pdf</p>
</div>
"""

with gr.Blocks(css=css) as app:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)

        with gr.Column():
            pdf_doc_input = gr.File(label="Load a pdf", file_types=['.pdf'], type="file")
            search_engine_input = gr.Radio(["FAISS", "Chroma"], label="Search engine", default="FAISS")
            split_method_input = gr.Radio(["CharacterTextSplitter", "RecursiveCharacterTextSplitter"], label="Split method", default="CharacterTextSplitter")

            with gr.Row():
                langchain_status = gr.Textbox(label="Status", placeholder="", interactive=False)
                load_pdf_button = gr.Button("Load pdf to langchain")

        chatbot = gr.Chatbot([], elem_id="chatbot").style(height=350)
        question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")
        submit_btn = gr.Button("Send Message")

    load_pdf_button.click(loading_pdf, None, langchain_status, queue=False)
    load_pdf_button.click(pdf_changes, inputs=[pdf_doc_input, search_engine_input, split_method_input], outputs=[langchain_status], queue=False)
    question.submit(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot
    )
    submit_btn.click(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot)

if __name__ == "__main__":
    if ENVIRONMENT == "development":
        print("Running in development mode")
        app.launch(debug=True, enable_queue=True)
    else:
        print("Running in production mode")
        app.launch(server_name="0.0.0.0", server_port=7860, enable_queue=True, auth=(USERNAME_LOGIN, PASSWORD_LOGIN))
