import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader

load_dotenv()

st.title("RAG with Groq LLM")


llm = ChatGroq(model="openai/gpt-oss-20b")

file = st.file_uploader("Upload a file", type=["pdf"])

if file is not None:    
    read = PdfReader(file)

    text = ""

    for page in read.pages:
        text += page.extract_text() or ""
        docs = [Document(page_content=text)]

    pdf = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    
    chunks = pdf.split_documents(docs)

   
    if len(chunks) > 0: 
    
        st.write(f"Number of chunks: {len(chunks)}")
        st.success("PDF uploaded sucessfully !")
    

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )   

    
        vectorstore = FAISS.from_documents(chunks,embeddings)

    
        retriever = vectorstore.as_retriever()

   
        question = st.text_input("Ask a question")

        if question:

            results = retriever.invoke(question)

            context = "\n".join(
                [doc.page_content for doc in results]
            )

            prompt = f"""
            Answer the question using the following context.
            Context: {context}
            Question:{question}"""

            response = llm.invoke(prompt)

            st.subheader("Answer for your question !")
            st.markdown(response.content)

else:
    
    st.warning("Please upload a file to proceed !")
    

