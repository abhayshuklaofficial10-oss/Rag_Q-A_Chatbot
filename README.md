👇

🧠 Custom Conversational RAG Q&A Chatbot

A conversational Retrieval-Augmented Generation (RAG) chatbot built with LangChain LCEL, Groq LLaMA 3, and Streamlit, designed to answer questions accurately from uploaded PDF documents while maintaining chat history and context awareness.

Unlike basic RAG systems, this chatbot supports multi-turn conversations by intelligently rephrasing follow-up questions into standalone queries before retrieval.

🚀 Key Features

📄 Multi-PDF Upload & Processing – Upload and chat with multiple PDF documents

🧠 Conversational Memory Support – Maintains chat history using LangChain message objects

🔄 Question Condensation – Converts follow-up questions into standalone queries for better retrieval

🔍 Semantic Search with Embeddings – Uses HuggingFace embeddings for accurate document retrieval

🗂️ Vector Store Integration – Stores and retrieves document chunks using Chroma DB

⚡ Fast LLM Inference – Powered by Groq’s LLaMA 3.1 for low-latency responses

🎯 Hallucination Control – Answers strictly from retrieved context or clearly states when the answer is unknown

🖥️ Interactive UI – Clean chat-based interface built with Streamlit

🛠️ Tech Stack

🧠 LLM: Groq – LLaMA 3.1 (8B Instant)

🔗 Framework: LangChain (LCEL-based pipelines)

📐 Embeddings: HuggingFace (all-MiniLM-L6-v2)

🗃️ Vector Database: Chroma

📄 Document Loader: PyPDFLoader

✂️ Text Splitting: RecursiveCharacterTextSplitter

🧪 Prompting: ChatPromptTemplate + Message Placeholders

🖥️ Frontend: Streamlit

🐍 Language: Python
