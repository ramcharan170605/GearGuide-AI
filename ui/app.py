import streamlit as st
import pandas as pd
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from assistant.agent import DealerAssistant
from assistant.retrieval import CatalogueRetriever
from assistant.tools import create_default_tool_registry

st.set_page_config(page_title='GearGuide-AI', page_icon='🚗', layout='wide')

st.markdown("""<style>
.stApp { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #ffffff; }
.header { text-align: center; padding: 20px 0; border-bottom: 1px solid rgba(255,255,255,0.1); }
.product-card { background: rgba(255,255,255,0.08); border-radius: 12px; padding: 16px; margin: 12px 0; border-left: 4px solid #00d4ff; }
.stock-in { background: #28a745; color: #fff; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold; display: inline-block; }
.stock-out { background: #dc3545; color: #fff; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold; display: inline-block; }
</style>""", unsafe_allow_html=True)

@st.cache_resource
def get_assistant():
    """Initialize and cache the assistant with LLM support."""
    # Use absolute paths based on project root for catalogue files
    catalogue = pd.read_csv(os.path.join(project_root, 'catalogue.csv'))
    retriever = CatalogueRetriever(os.path.join(project_root, 'catalogue.csv'))
    retriever.load_catalogue()
    retriever.initialize_embedding_model()
    retriever.build_vector_store()
    tool_registry = create_default_tool_registry(catalogue)
    return DealerAssistant(retriever, tool_registry)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'assistant' not in st.session_state:
    st.session_state.assistant = get_assistant()
if 'llm_available' not in st.session_state:
    st.session_state.llm_available = st.session_state.assistant._llm_available

# Header
st.markdown('<div class="header"><h1 style="color:#00d4ff;margin:0;">🚗 GearGuide-AI</h1><p style="color:#8892b0;margin:10px 0 0 0;">LLM-Powered Dealer Assistant</p></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('## ☰ Menu')
    catalogue = pd.read_csv(os.path.join(project_root, 'catalogue.csv'))
    st.metric('Products', f'{len(catalogue):,}')
    st.metric('Categories', catalogue['category'].nunique())
    st.metric('In Stock', f'{catalogue[catalogue["stock"] > 0].shape[0]:,}')

    # Show LLM status
    if st.session_state.llm_available:
        st.success("✓ LLM Available")
    else:
        st.warning("⚠ LLM Not Available (using fallback)")

    st.markdown('### ⚡ Quick Actions')
    quick_queries = [
        'Show me brake pads',
        "What's the stock of BRK-1007?",
        'Find parts for Bajaj Pulsar 150',
        'Place an order for 5 units of OIL-1001',
        'I need tyres',
    ]
    for query in quick_queries:
        if st.button(query, use_container_width=True):
            st.session_state.messages = []
            st.session_state.messages.append({'role': 'user', 'content': query})
            with st.spinner("Processing..."):
                response = st.session_state.assistant.process_query(query)
            st.session_state.messages.append({'role': 'assistant', 'content': response})
            st.rerun()

    st.markdown("---")

    # Clear conversation button
    if st.button('🗑️ Clear Conversation', use_container_width=True):
        st.session_state.messages = []
        st.session_state.assistant = get_assistant()  # Reset assistant
        st.rerun()

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Chat input
if prompt := st.chat_input('Ask about auto parts...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)
    with st.chat_message('assistant'):
        with st.spinner("Thinking..."):
            response = st.session_state.assistant.process_query(prompt)
        st.markdown(response)
    st.session_state.messages.append({'role': 'assistant', 'content': response})
