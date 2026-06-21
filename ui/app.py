import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    catalogue = pd.read_csv('catalogue.csv')
    retriever = CatalogueRetriever('catalogue.csv')
    retriever.load_catalogue()
    retriever.initialize_embedding_model()
    retriever.build_vector_store()
    tool_registry = create_default_tool_registry(catalogue)
    return DealerAssistant(retriever, tool_registry)

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'assistant' not in st.session_state:
    st.session_state.assistant = get_assistant()

st.markdown('<div class="header"><h1 style="color:#00d4ff;margin:0;">🚗 GearGuide-AI</h1><p style="color:#8892b0;margin:10px 0 0 0;">Premium Dealer Assistant</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('## ☰ Menu')
    catalogue = pd.read_csv('catalogue.csv')
    st.metric('Products', f'{len(catalogue):,}')
    st.metric('Categories', catalogue['category'].nunique())
    st.metric('In Stock', f'{catalogue[catalogue["stock"] > 0].shape[0]:,}')
    st.markdown('### ⚡ Quick Actions')
    quick_queries = [
        'Show me brake pads',
        "What's the stock of BRK-1007?",
        'Find parts for Bajaj Pulsar 150',
        'Place an order for 5 units of OIL-1001',
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

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if prompt := st.chat_input('Ask about auto parts...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)
    with st.chat_message('assistant'):
        with st.spinner("Thinking..."):
            response = st.session_state.assistant.process_query(prompt)
        st.markdown(response)
    st.session_state.messages.append({'role': 'assistant', 'content': response})