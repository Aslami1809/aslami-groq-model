# Title: ASLAMI RAG Chatbot Using LangGraph
# Description: Premium elegant UI chatbot using LangGraph, Groq's llama3-8b-8192 model, and Streamlit

# Import required packages
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from typing import TypedDict, List
from langchain_core.messages import BaseMessage
import time

# Setup model
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.5,
    api_key="GROQ_API_KEY",  # Replace with actual key
    streaming=True
)

# Define state structure
class State(TypedDict):
    messages: List[BaseMessage]

# Define node for LangGraph
def generate_response(state: State):
    try:
        human_input = state["messages"][-1].content
        response = llm.invoke([HumanMessage(content=human_input)])
        return {"messages": state["messages"] + [AIMessage(content=response.content)]}
    except Exception as e:
        error_msg = f"I apologize, but I encountered an issue processing your request. Please try again."
        return {"messages": state["messages"] + [AIMessage(content=error_msg)]}

# Build the graph
builder = StateGraph(State)
builder.add_node("chat", generate_response)
builder.set_entry_point("chat")
builder.set_finish_point("chat")

graph = builder.compile()

# Page configuration
st.set_page_config(
    page_title="ASLAMI AI - Premium Conversational Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Global dark theme */
    .stApp {
        background: #1a1a1a;
        color: #ffffff;
    }
    
    /* Streamlit elements styling */
    .stMarkdown {
        color: #ffffff;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ffffff;
    }
    
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
        max-width: 1200px;
        margin: 0 auto;
        background: #1a1a1a;
        color: #ffffff;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #2c3e50 0%, #1a252f 100%);
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        border-bottom: 1px solid #34495e;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 300;
        color: #ffffff;
        text-align: center;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #bdc3c7;
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Chat container */
    .chat-container {
        background: #2c2c2c;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.15);
        padding: 1.5rem;
        margin-bottom: 2rem;
        border: 1px solid #404040;
    }
    
    /* Message styling */
    .user-message {
        background: #404040;
        padding: 1rem 1.25rem;
        border-radius: 18px 18px 4px 18px;
        margin: 1rem 0 1rem 2rem;
        border-left: 3px solid #2980b9;
        font-size: 0.95rem;
        line-height: 1.5;
        color: #ffffff;
    }
    
    .ai-message {
        background: #2c2c2c;
        padding: 1rem 1.25rem;
        border-radius: 18px 18px 18px 4px;
        margin: 1rem 2rem 1rem 0;
        border: 1px solid #404040;
        border-left: 3px solid #34495e;
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
        color: #ffffff;
    }
    
    .message-label {
        font-weight: 600;
        font-size: 0.85rem;
        color: #bdc3c7;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #404040;
        padding: 0.75rem 1.25rem;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        background: #2c2c2c;
        color: #ffffff;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #2980b9;
        box-shadow: 0 0 0 3px rgba(41, 128, 185, 0.2);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #2980b9 0%, #34495e 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.6rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
        background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%);
    }
    
    /* Features section */
    .features-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: #2c2c2c;
        border: 1px solid #404040;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        border-color: #2980b9;
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        font-size: 0.9rem;
        color: #bdc3c7;
        line-height: 1.5;
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #bdc3c7;
        font-style: italic;
        margin: 1rem 0;
    }
    
    .typing-dots {
        display: flex;
        gap: 2px;
    }
    
    .typing-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #2980b9;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes typing {
        0%, 80%, 100% { opacity: 0.3; }
        40% { opacity: 1; }
    }
    
    /* Professional footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #bdc3c7;
        font-size: 0.9rem;
        border-top: 1px solid #404040;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown("""
<div class="header-container">
    <h1 class="main-title">ASLAMI</h1>
    <p class="subtitle">Advanced AI Conversational Assistant | Powered by LangGraph & Groq</p>
</div>
""", unsafe_allow_html=True)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "is_typing" not in st.session_state:
    st.session_state.is_typing = False

# Main chat interface
col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    # Input section
    st.markdown("### Start Your Conversation")
    user_input = st.text_input(
        "",
        placeholder="Ask me anything... I'm here to help you with intelligent responses",
        key="input",
        label_visibility="collapsed"
    )
    
    # Process user input
    if user_input and not st.session_state.is_typing:
        st.session_state.is_typing = True
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        
        try:
            # Show typing indicator
            with st.empty():
                st.markdown("""
                <div class="typing-indicator">
                    <span>ASLAMI is thinking</span>
                    <div class="typing-dots">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(1)
            
            result = graph.invoke({"messages": st.session_state.chat_history})
            st.session_state.chat_history = result["messages"]
            
        except Exception as e:
            st.error("I encountered an issue. Please try again or contact support.")
        finally:
            st.session_state.is_typing = False
            st.rerun()

    # Display conversation
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### Conversation History")
        
        for i, msg in enumerate(st.session_state.chat_history):
            if isinstance(msg, HumanMessage):
                st.markdown(f"""
                <div class="user-message">
                    <div class="message-label">You</div>
                    {msg.content}
                </div>
                """, unsafe_allow_html=True)
            elif isinstance(msg, AIMessage):
                st.markdown(f"""
                <div class="ai-message">
                    <div class="message-label">ASLAMI Assistant</div>
                    {msg.content}
                </div>
                """, unsafe_allow_html=True)
    
    # Action buttons
    if st.session_state.chat_history:
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            if st.button("🗑️ Clear Conversation", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()

# Features section (only show when no conversation)
if not st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### Why Choose ASLAMI?")
    
    st.markdown("""
    <div class="features-container">
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">Advanced AI Intelligence</div>
            <div class="feature-desc">Powered by cutting-edge language models for sophisticated conversations and problem-solving</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Lightning Fast Responses</div>
            <div class="feature-desc">Get instant, accurate answers with our optimized processing pipeline</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔒</div>
            <div class="feature-title">Enterprise Security</div>
            <div class="feature-desc">Your conversations are protected with industry-standard security protocols</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Context-Aware</div>
            <div class="feature-desc">Maintains conversation context for more meaningful and relevant interactions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Professional footer
st.markdown("""
<div class="footer">
    <p><strong>ASLAMI AI Assistant</strong> | Professional Grade Conversational AI</p>
    <p>Experience the future of intelligent conversation</p>
</div>
""", unsafe_allow_html=True)