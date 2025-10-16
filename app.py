"""
Advanced PDF RAG System with Image Descriptions
Streamlit application for PDF processing and question answering
"""
import streamlit as st
import os
import tempfile
from pathlib import Path
from PIL import Image
from datetime import datetime

from utils.pdf_processor import PDFProcessor
from utils.image_describer import ImageDescriber
from utils.vector_store import VectorStore
from utils.rag_engine import RAGEngine
from utils.pdf_cache import PDFCache

try:
    from config import Config
    USE_CONFIG = True
except ImportError:
    USE_CONFIG = False
    Config = None


# Page configuration
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional UI
def load_custom_css():
    st.markdown("""
    <style>
    /* Main color scheme */
    :root {
        --primary-color: #6366f1;
        --primary-hover: #4f46e5;
        --bg-color: #0f172a;
        --sidebar-bg: #1e293b;
        --chat-bg: #f8fafc;
        --user-msg-bg: #6366f1;
        --assistant-msg-bg: #ffffff;
        --border-color: #e2e8f0;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        padding: 1.5rem 1rem;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
        color: #cbd5e1 !important;
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
        border: none;
        padding: 0.5rem 1rem;
    }
    
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
    }
    
    .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
        transform: translateY(-1px);
    }
    
    .stButton button[kind="secondary"] {
        background: #f1f5f9;
        color: #475569;
        border: 1px solid #e2e8f0;
    }
    
    .stButton button[kind="secondary"]:hover {
        background: #e2e8f0;
        border-color: #cbd5e1;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    [data-testid="stChatMessageContent"] {
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* User message */
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        margin-left: 20%;
    }
    
    /* Assistant message */
    .stChatMessage[data-testid="assistant-message"] {
        background: white;
        border: 1px solid #e2e8f0;
        margin-right: 20%;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(99, 102, 241, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        border: 2px dashed #6366f1;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader:hover {
        background: #f1f5f9;
        border-color: #cbd5e1;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    
    /* Info/Success/Warning boxes */
    .stAlert {
        border-radius: 10px;
        border-left-width: 4px;
        padding: 1rem 1.25rem;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        border-radius: 10px;
    }
    
    /* Input fields */
    .stTextInput input, .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }
    
    /* Chat input */
    .stChatInput {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
    }
    
    .stChatInput:focus-within {
        border-color: #6366f1;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }
    
    /* Divider */
    hr {
        margin: 1.5rem 0;
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    
    /* Title styling */
    h1 {
        color: #1e293b;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    h2, h3 {
        color: #334155;
        font-weight: 600;
    }
    
    /* Caption styling */
    .caption {
        color: #64748b;
        font-size: 0.875rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "processed" not in st.session_state:
    st.session_state.processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "pdf_cache" not in st.session_state:
    st.session_state.pdf_cache = PDFCache()


def initialize_system():
    """Initialize the RAG system components"""
    if st.session_state.vector_store is None:
        st.session_state.vector_store = VectorStore()
    if st.session_state.rag_engine is None:
        st.session_state.rag_engine = RAGEngine(st.session_state.vector_store)


def process_pdf(pdf_file):
    """Process uploaded PDF file (with caching)"""
    pdf_name = pdf_file.name.replace(".pdf", "")
    
    # FIRST: Check if already in vector store (fastest check)
    if st.session_state.vector_store.is_pdf_processed(pdf_name):
        with st.spinner("📂 Loading existing PDF data..."):
            st.info("✅ PDF already processed! Using existing data from vector database.")
            st.session_state.processed = True
            st.session_state.pdf_name = pdf_name
            
            # Show stats
            stats = st.session_state.vector_store.get_stats()
            st.success(f"📊 Loaded: {stats['total_chunks']} text chunks available for querying")
            return
    
    # PDF not in vector store, need to process
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name
    
    try:
        
        # Process PDF (not cached or not in vector store)
        with st.spinner("Processing PDF..."):
            # Initialize processors with GPU if available
            use_gpu = Config.USE_GPU if USE_CONFIG else False
            pdf_processor = PDFProcessor(use_gpu=use_gpu)
            image_describer = ImageDescriber()
            
            # Process PDF
            progress_bar = st.progress(0)
            st.info("Step 1/5: Extracting text and images from PDF...")
            progress_bar.progress(20)
            
            processed_data = pdf_processor.process_pdf(tmp_path, pdf_name)
            markdown_content = processed_data.get('markdown', '')
            
            st.info(f"Step 2/5: Found {len(processed_data['images'])} images. Generating descriptions...")
            progress_bar.progress(40)
            
            # Describe images
            described_images = image_describer.describe_images_batch(processed_data['images'])
            
            st.info("Step 3/5: Enriching pages with image descriptions...")
            progress_bar.progress(50)
            
            # Enrich markdown and pages with image descriptions
            enriched_pages, enriched_markdown = pdf_processor.enrich_with_image_descriptions(
                processed_data['pages'],
                processed_data['markdown'],
                described_images
            )
            
            # Update processed_data with enriched content
            processed_data['pages'] = enriched_pages
            processed_data['markdown'] = enriched_markdown
            
            st.info("Step 4/5: Creating vector embeddings...")
            progress_bar.progress(65)
            
            # Store in vector database (now with enriched content)
            st.session_state.vector_store.add_documents(processed_data, described_images)
            
            st.info("Step 5/5: Caching processed data...")
            progress_bar.progress(85)
            
            # Cache the processed data (enriched markdown)
            st.session_state.pdf_cache.save_to_cache(
                tmp_path, 
                pdf_name, 
                processed_data, 
                enriched_markdown  # Save enriched markdown
            )
            
            st.info("Step 5/5: Finalizing...")
            progress_bar.progress(100)
            
            # Update session state
            st.session_state.processed = True
            st.session_state.pdf_name = pdf_name
            
            # Clean up
            os.unlink(tmp_path)
            
            st.success(f"✅ PDF processed and cached! Found {len(processed_data['pages'])} pages and {len(described_images)} images.")
            
            # Show stats
            stats = st.session_state.vector_store.get_stats()
            st.info(f"📊 Vector database contains {stats['total_chunks']} text chunks")
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def display_chat_message(message):
    """Display a chat message with citations and images"""
    role = message.get("role", "user")
    content = message.get("content", "")
    
    with st.chat_message(role):
        st.write(content)
        
        # Display citations if available
        if role == "assistant" and "citations" in message:
            citations = message["citations"]
            model_used = message.get("model_used", "Unknown")
            
            if citations:
                st.markdown("---")
                st.caption(f"🤖 **Model Used:** {model_used}")
                st.caption("📑 **Sources & Citations:**")
                
                for citation in citations:
                    with st.expander(f"📄 Page {citation['page']} - Source {citation['source_id']}"):
                        st.write(f"**Text Snippet:** {citation['text_snippet']}")
                        
                        # Display images if available
                        if citation['has_images'] and citation['image_paths']:
                            st.write("**Related Images:**")
                            cols = st.columns(min(len(citation['image_paths']), 3))
                            for idx, img_path in enumerate(citation['image_paths']):
                                if os.path.exists(img_path):
                                    with cols[idx % 3]:
                                        image = Image.open(img_path)
                                        st.image(image, use_container_width=True)


def main():
    """Main Streamlit application"""
    st.title("📚 Advanced PDF RAG System")
    st.markdown("Upload a PDF and ask questions with AI-powered image understanding")
    
    # Show API and GPU status if config available
    if USE_CONFIG and Config:
        status = Config.get_api_status()
        col1, col2 = st.columns(2)
        with col1:
            if status['hf_api_configured']:
                st.success("🚀 HuggingFace API")
            else:
                st.warning("⚠️ API Not Configured")
        with col2:
            if status['gpu_available']:
                st.success(f"⚡ GPU: {status['gpu_name'][:20]}")
            else:
                st.info("💻 CPU Mode")
    
    # Initialize system
    initialize_system()
    
    # Sidebar
    with st.sidebar:
        st.header("📁 PDF Upload")
        
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        
        if uploaded_file:
            # Check if this PDF is already processed
            pdf_name_check = uploaded_file.name.replace(".pdf", "")
            is_already_processed = st.session_state.vector_store.is_pdf_processed(pdf_name_check)
            
            if is_already_processed:
                st.info("📌 This PDF is already in the database")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📂 Load Existing", type="primary", use_container_width=True):
                        st.session_state.processed = True
                        st.session_state.pdf_name = pdf_name_check
                        st.session_state.chat_history = []
                        st.rerun()
                with col2:
                    if st.button("🔄 Reprocess", use_container_width=True):
                        # Clear old data and reprocess
                        st.session_state.vector_store.clear_collection()
                        st.session_state.chat_history = []
                        process_pdf(uploaded_file)
            else:
                if st.button("🚀 Process PDF", type="primary", use_container_width=True):
                    # Clear previous data if different PDF
                    if st.session_state.processed and st.session_state.pdf_name != pdf_name_check:
                        st.session_state.chat_history = []
                    
                    process_pdf(uploaded_file)
        
        st.markdown("---")
        
        # System info
        if st.session_state.processed:
            st.success("✅ System Ready")
            st.write(f"**Current PDF:** {st.session_state.pdf_name}")
            
            stats = st.session_state.vector_store.get_stats()
            st.metric("Text Chunks", stats.get('total_chunks', 0))
            
            # Show cached PDFs
            cache_info = st.session_state.pdf_cache.get_cache_info()
            if cache_info['total_cached'] > 0:
                with st.expander(f"📂 Cached PDFs ({cache_info['total_cached']})"):
                    for pdf in cache_info['pdfs']:
                        st.caption(f"• {pdf['name']} ({pdf['pages']} pages, {pdf['images']} images)")
        else:
            st.info("⏳ Upload and process a PDF to start")
            
            # Show what's already in database
            processed_pdfs = st.session_state.vector_store.get_processed_pdfs()
            if processed_pdfs:
                st.info(f"📚 {len(processed_pdfs)} PDF(s) in database")
                with st.expander("View processed PDFs"):
                    for pdf in processed_pdfs:
                        st.caption(f"• {pdf}")
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        # Reset system button
        if st.button("🔄 Reset System", use_container_width=True):
            st.session_state.vector_store.clear_collection()
            st.session_state.pdf_cache.clear_cache()
            st.session_state.chat_history = []
            st.session_state.processed = False
            st.session_state.pdf_name = None
            st.rerun()
        
        st.markdown("---")
        st.caption("**Technologies:**")
        st.caption("• Docling (PDF → Markdown)")
        st.caption("• Qwen2.5-VL-7B (Image AI)")
        st.caption("• Qwen2.5-7B (Chat AI)")
        st.caption("• MiniLM-L6 (Local Embeddings)")
        st.caption("• ChromaDB (Vector Store)")
        st.caption("• HuggingFace API (Image & Chat)")
    
    # Main chat interface
    if st.session_state.processed:
        # Display chat history
        for message in st.session_state.chat_history:
            display_chat_message(message)
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your PDF..."):
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": prompt
            })
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.rag_engine.chat_with_history(
                        prompt,
                        st.session_state.chat_history,
                        n_results=5
                    )
                    
                    # Display answer
                    st.write(response["answer"])
                    
                    # Display citations
                    citations = response["citations"]
                    model_used = response["model_used"]
                    
                    if citations:
                        st.markdown("---")
                        st.caption(f"🤖 **Model Used:** {model_used}")
                        st.caption("📑 **Sources & Citations:**")
                        
                        for citation in citations:
                            with st.expander(f"📄 Page {citation['page']} - Source {citation['source_id']}"):
                                st.write(f"**Text Snippet:** {citation['text_snippet']}")
                                
                                # Display images if available
                                if citation['has_images'] and citation['image_paths']:
                                    st.write("**Related Images:**")
                                    cols = st.columns(min(len(citation['image_paths']), 3))
                                    for idx, img_path in enumerate(citation['image_paths']):
                                        if os.path.exists(img_path):
                                            with cols[idx % 3]:
                                                image = Image.open(img_path)
                                                st.image(image, use_container_width=True)
                    
                    # Add assistant message to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "citations": citations,
                        "model_used": model_used
                    })
    
    else:
        # Welcome screen
        st.info("👈 Please upload and process a PDF file from the sidebar to get started")
        
        st.markdown("### ✨ Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Document Processing:**
            - 📄 PDF to Markdown conversion
            - 🖼️ Automatic image extraction
            - 🤖 AI-powered image descriptions
            - 📊 Intelligent text chunking
            """)
        
        with col2:
            st.markdown("""
            **Smart Q&A:**
            - 💬 Natural language queries
            - 🎯 Context-aware responses
            - 📍 Page-level citations
            - 🖼️ Image context in answers
            """)
        
        st.markdown("### 🚀 How to Use")
        st.markdown("""
        1. **Upload** your PDF file using the sidebar
        2. Click **Process PDF** to extract text and images
        3. **Ask questions** about the content
        4. Get **AI-powered answers** with citations and relevant images
        """)


if __name__ == "__main__":
    main()

