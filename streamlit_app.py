import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

st.set_page_config(page_title="HR Assistant - Switzerland", layout="centered")
st.title("🇨🇭 HR Assistant for Novocure (Switzerland)")

@st.cache_resource
def load_qa():
    loader = PyPDFLoader("Switzerland_Novocure_GEP.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory="./vector_db"
    )

    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)
    return qa_chain

qa_chain = load_qa()

query = st.text_input("Ask your HR question:")

if query:
    with st.spinner("Searching the HR policy..."):
        answer = qa_chain.run(query)
    st.markdown("### ✅ Answer")
    st.write(answer)
