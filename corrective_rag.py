import streamlit as st
import os
import tempfile
from typing import List, Optional
import PyPDF2
import docx
from io import BytesIO
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.embeddings.base import Embeddings

# Groq imports
from langchain_groq import ChatGroq

# Try different embedding options
try:
    from langchain.embeddings import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class TfidfEmbeddings(Embeddings):
    """Custom TF-IDF based embeddings as fallback"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
        self.is_fitted = False
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.is_fitted:
            self.vectorizer.fit(texts)
            self.is_fitted = True
            
        embeddings = self.vectorizer.transform(texts).toarray()
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        if not self.is_fitted:
            # If not fitted, return zero vector
            return [0.0] * 384
        embedding = self.vectorizer.transform([text]).toarray()[0]
        return embedding.tolist()

class DocumentProcessor:
    """Handle document loading and processing"""
    
    @staticmethod
    def extract_text_from_pdf(file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return ""
    
    @staticmethod
    def extract_text_from_txt(file) -> str:
        """Extract text from TXT file"""
        try:
            return file.read().decode('utf-8')
        except Exception as e:
            st.error(f"Error reading TXT: {str(e)}")
            return ""
    
    def process_uploaded_files(self, uploaded_files) -> List[Document]:
        """Process uploaded files and return list of Document objects"""
        documents = []
        
        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'pdf':
                text = self.extract_text_from_pdf(uploaded_file)
            elif file_extension == 'docx':
                text = self.extract_text_from_docx(uploaded_file)
            elif file_extension == 'txt':
                text = self.extract_text_from_txt(uploaded_file)
            else:
                st.warning(f"Unsupported file format: {uploaded_file.name}")
                continue
            
            if text.strip():
                doc = Document(
                    page_content=text,
                    metadata={"source": uploaded_file.name}
                )
                documents.append(doc)
        
        return documents

class RAGChatbot:
    """Main RAG Chatbot class with multiple embedding options"""
    
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.retrieval_chain = None
        self.memory = None
        self.document_processor = DocumentProcessor()
        self.embedding_method = None
        
    def initialize_embeddings(self, method='auto'):
        """Initialize embedding model with fallback options"""
        
        if method == 'auto':
            # Try methods in order of preference
            methods_to_try = ['huggingface_offline', 'sentence_transformers', 'huggingface_online', 'tfidf']
        else:
            methods_to_try = [method]
        
        for method in methods_to_try:
            try:
                st.info(f"Attempting to initialize embeddings using: {method}")
                
                if method == 'huggingface_online' and HUGGINGFACE_AVAILABLE:
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs={'device': 'cpu'}
                    )
                    self.embedding_method = 'HuggingFace Online'
                    st.success("✅ Successfully initialized HuggingFace embeddings (online)")
                    return True
                    
                elif method == 'huggingface_offline' and HUGGINGFACE_AVAILABLE:
                    # Try to use local cache
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs={'device': 'cpu', 'local_files_only': True}
                    )
                    self.embedding_method = 'HuggingFace Offline'
                    st.success("✅ Successfully initialized HuggingFace embeddings (offline)")
                    return True
                    
                elif method == 'sentence_transformers' and SENTENCE_TRANSFORMERS_AVAILABLE:
                    # Try sentence-transformers directly
                    model = SentenceTransformer('all-MiniLM-L6-v2')
                    
                    class SentenceTransformerEmbeddings(Embeddings):
                        def __init__(self, model):
                            self.model = model
                        
                        def embed_documents(self, texts: List[str]) -> List[List[float]]:
                            return self.model.encode(texts).tolist()
                        
                        def embed_query(self, text: str) -> List[float]:
                            return self.model.encode([text])[0].tolist()
                    
                    self.embeddings = SentenceTransformerEmbeddings(model)
                    self.embedding_method = 'SentenceTransformers'
                    st.success("✅ Successfully initialized SentenceTransformers embeddings")
                    return True
                    
                elif method == 'tfidf':
                    self.embeddings = TfidfEmbeddings()
                    self.embedding_method = 'TF-IDF'
                    st.success("✅ Successfully initialized TF-IDF embeddings (offline)")
                    return True
                    
            except Exception as e:
                st.warning(f"Failed to initialize {method}: {str(e)}")
                continue
        
        st.error("❌ Failed to initialize any embedding method")
        return False
    
    def initialize_llm(self):
        """Initialize Groq LLM"""
        try:
            self.llm = ChatGroq(
                groq_api_key=self.groq_api_key,
                model_name="llama3-8b-8192",  # or "llama3-70b-8192" for larger model
                temperature=0.7,
                max_tokens=1024
            )
            return True
        except Exception as e:
            st.error(f"Error initializing LLM: {str(e)}")
            return False
    
    def create_vectorstore(self, documents: List[Document]):
        """Create vector store from documents"""
        try:
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            chunks = text_splitter.split_documents(documents)
            
            if not chunks:
                st.error("No text chunks created from documents")
                return False
            
            # Create vector store
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            st.success(f"✅ Created vector store with {len(chunks)} chunks using {self.embedding_method}")
            return True
            
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return False
    
    def setup_retrieval_chain(self):
        """Setup the conversational retrieval chain"""
        try:
            if not all([self.llm, self.vectorstore]):
                st.error("LLM or vectorstore not initialized")
                return False
            
            # Create custom prompt template
            prompt_template = """You are a helpful assistant that answers questions based on the provided context. 
            Use the following context to answer the question. If you cannot find the answer in the context, 
            say "I don't have enough information in the provided documents to answer this question."

            Context: {context}

            Chat History: {chat_history}

            Question: {question}

            Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "chat_history", "question"]
            )
            
            # Initialize memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            # Create retrieval chain
            self.retrieval_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": PROMPT},
                return_source_documents=True,
                verbose=True
            )
            
            return True
            
        except Exception as e:
            st.error(f"Error setting up retrieval chain: {str(e)}")
            return False
    
    def query(self, question: str) -> dict:
        """Query the RAG system"""
        try:
            if not self.retrieval_chain:
                return {"error": "Retrieval chain not initialized"}
            
            response = self.retrieval_chain({"question": question})
            return response
            
        except Exception as e:
            return {"error": f"Error during query: {str(e)}"}

def main():
    st.set_page_config(
        page_title="AslamiGroq RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AslamiGroq RAG Chatbot")
    st.markdown("Upload documents and chat with them using Groq's Llama model!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Enter your Groq API key"
        )
        
        if not groq_api_key:
            st.warning("Please enter your Groq API key to continue")
            st.info("Get your API key from: https://console.groq.com/keys")
            return
        
        st.success("API Key provided ✅")
        
        # Embedding method selection
        st.header("🔧 Embedding Settings")
        embedding_method = st.selectbox(
            "Choose embedding method:",
            ["auto", "tfidf", "huggingface_offline", "sentence_transformers", "huggingface_online"],
            help="Auto will try methods in order of preference. TF-IDF works offline."
        )
        
        # File upload
        st.header("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx'],
            help="Upload PDF, TXT, or DOCX files (max 200MB per file)"
        )
        
        # Process documents button
        if uploaded_files and st.button("🔄 Process Documents"):
            with st.spinner("Processing documents..."):
                # Initialize chatbot
                chatbot = RAGChatbot(groq_api_key)
                
                # Initialize components
                if not chatbot.initialize_embeddings(method=embedding_method):
                    st.error("Failed to initialize embeddings")
                    return
                
                if not chatbot.initialize_llm():
                    st.error("Failed to initialize LLM")
                    return
                
                # Process documents
                documents = chatbot.document_processor.process_uploaded_files(uploaded_files)
                
                if not documents:
                    st.error("No valid documents found")
                    return
                
                # Create vector store
                if not chatbot.create_vectorstore(documents):
                    st.error("Failed to create vector store")
                    return
                
                # Setup retrieval chain
                if not chatbot.setup_retrieval_chain():
                    st.error("Failed to setup retrieval chain")
                    return
                
                # Store chatbot in session state
                st.session_state.chatbot = chatbot
                st.session_state.documents_processed = True
                
                st.success(f"✅ Processed {len(documents)} documents successfully using {chatbot.embedding_method}!")
    
    # Main chat interface
    if 'documents_processed' in st.session_state and st.session_state.documents_processed:
        st.header("💬 Chat with your documents")
        
        # Show current embedding method
        if hasattr(st.session_state.chatbot, 'embedding_method'):
            st.info(f"Using embeddings: {st.session_state.chatbot.embedding_method}")
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.write(f"• {source}")
        
        # Chat input
        if question := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": question})
            
            # Display user message
            with st.chat_message("user"):
                st.write(question)
            
            # Get response from chatbot
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chatbot.query(question)
                    
                    if "error" in response:
                        st.error(response["error"])
                    else:
                        answer = response.get("answer", "I couldn't generate an answer.")
                        st.write(answer)
                        
                        # Show sources if available
                        sources = []
                        if "source_documents" in response:
                            sources = [doc.metadata.get("source", "Unknown") 
                                     for doc in response["source_documents"]]
                            
                            if sources:
                                with st.expander("📚 Sources"):
                                    for source in set(sources):  # Remove duplicates
                                        st.write(f"• {source}")
                        
                        # Add assistant message to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": list(set(sources))
                        })
    
    else:
        st.info("👆 Please upload and process documents to start chatting!")
        
        # Show embedding options info
        st.header("🔧 Embedding Options")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🌐 Online Methods
            - **HuggingFace Online**: Best quality, requires internet
            - **SentenceTransformers**: Good quality, may require download
            """)
        
        with col2:
            st.markdown("""
            ### 💻 Offline Methods  
            - **TF-IDF**: Works completely offline, decent quality
            - **HuggingFace Offline**: Uses cached models if available
            """)
        
        # Show features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 📄 Document Support
            - PDF files
            - Word documents (.docx)
            - Text files (.txt)
            """)
        
        with col2:
            st.markdown("""
            ### 🧠 Powered by
            - Groq Llama models
            - Multiple embedding options
            - Vector search
            """)
        
        with col3:
            st.markdown("""
            ### ✨ Features
            - Offline capability
            - Conversational memory
            - Source citations
            """)

if __name__ == "__main__":
    main()
