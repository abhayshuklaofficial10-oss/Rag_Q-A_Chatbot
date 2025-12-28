import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Custom RAG", layout="wide")
st.title("🧠 Custom Conversational RAG Chatbot")

# --- Session State Initialization ---
# Ensure variables are stored in st.session_state so they persist across reruns
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- Sidebar: API Key and File Upload ---
with st.sidebar:
    api_key = st.text_input("Groq API Key", type="password")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    process_btn = st.button("Process Documents")

# --- Document Processing ---
if process_btn and uploaded_files and api_key:
    with st.spinner("Indexing documents..."):
        documents = []
        for file in uploaded_files:
            # Save uploaded file temporarily for the loader
            temp_path = f"./{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            
            loader = PyPDFLoader(temp_path)
            documents.extend(loader.load())
            os.remove(temp_path) # Cleanup temporary file

        # Split documents into smaller chunks for retrieval
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(documents)
        
        # Initialize embeddings and store in session state
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        st.success("Indexing Complete!")

# --- Main Chat Logic ---
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

    # 1. Standalone Question Prompt (Condensing History)
    # This turns "How does it work?" into "How does the Transformer work?"
    condense_system_prompt = """Given a chat history and the latest user question 
    which might reference context in the chat history, formulate a standalone question 
    which can be understood without the chat history. Do NOT answer the question."""
    
    condense_prompt = ChatPromptTemplate.from_messages([
        ("system", condense_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # 2. Final Answer Prompt
    qa_system_prompt = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question.
    If the context doesn't contain the answer, say that you don't know. 
    
    Context:
    {context}"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # --- LCEL Chain Construction ---
    if st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever()

        # Chain A: Create Standalone Question
        condense_chain = condense_prompt | llm | StrOutputParser()

        # Chain B: Final Logic (Condense -> Retrieve -> Answer)
        def rag_logic(user_dict):
            # Step 1: Rephrase question using history
            standalone_query = condense_chain.invoke({
                "chat_history": st.session_state.chat_history,
                "input": user_dict["input"]
            })
            
            # Step 2: Retrieve relevant chunks using .invoke() instead of .get_relevant_documents()
            docs = retriever.invoke(standalone_query) # FIX APPLIED HERE
            context_text = "\n\n".join([d.page_content for d in docs])
            
            # Step 3: Pass context and history to the final answer generation
            return qa_prompt.invoke({
                "context": context_text,
                "chat_history": st.session_state.chat_history,
                "input": user_dict["input"]
            })

        # Define the complete runnable sequence
        full_chain = RunnablePassthrough() | rag_logic | llm | StrOutputParser()

        # --- UI Chat Interface ---
        user_input = st.chat_input("Ask something about your PDF...")

        if user_input:
            # Display current chat history from session state
            for message in st.session_state.chat_history:
                role = "user" if isinstance(message, HumanMessage) else "assistant"
                with st.chat_message(role):
                    st.markdown(message.content)

            # Display the new user message
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                try:
                    response = full_chain.invoke({"input": user_input})
                    st.markdown(response)
                    
                    # Store interaction in session history
                    st.session_state.chat_history.append(HumanMessage(content=user_input))
                    st.session_state.chat_history.append(AIMessage(content=response))
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("Please upload and process a PDF to start chatting.")
else:
    st.warning("Please enter your Groq API Key in the sidebar.")