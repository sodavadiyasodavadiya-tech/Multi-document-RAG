import os
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from vectordb import VectorStore
from rag_pipeline import build_index, answer_multiple_questions

# ===================== CONFIG ===================== #
st.set_page_config(
    page_title="RAG Chat - Gemini 2.5",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== STYLING ===================== #
st.markdown("""
<style>
    /* Modern Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Main Title Styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 5px rgba(102, 126, 234, 0.5)); }
        to { filter: drop-shadow(0 0 20px rgba(118, 75, 162, 0.8)); }
    }
    
    .subtitle {
        text-align: center;
        color: #a0aec0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    /* Chat Messages */
    .question-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    
    .answer-bubble {
        background: rgba(255, 255, 255, 0.08);
        color: #e2e8f0;
        padding: 1.5rem;
        border-radius: 5px 20px 20px 20px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Input Styling */
    .stTextInput>div>div>input {
        background: rgba(255, 255, 255, 0.05);
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: white;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
    }
    
    /* File Uploader */
    .stFileUploader>div>div {
        background: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 15px;
        transition: all 0.3s ease;
    }
    
    .stFileUploader>div>div:hover {
        border-color: #764ba2;
        background: rgba(102, 126, 234, 0.1);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: rgba(15, 12, 41, 0.95);
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: rgba(72, 187, 120, 0.1);
        border: 1px solid rgba(72, 187, 120, 0.3);
        border-radius: 12px;
    }
    
    .stError {
        background: rgba(245, 101, 101, 0.1);
        border: 1px solid rgba(245, 101, 101, 0.3);
        border-radius: 12px;
    }
    
    /* Spinner */
    .stSpinner>div {
        border-color: #667eea !important;
    }
    
    /* Source Tags */
    .source-tag {
        display: inline-block;
        background: rgba(102, 126, 234, 0.2);
        color: #667eea;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 0.25rem;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Stats Cards */
    .stat-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .stat-label {
        color: #a0aec0;
        font-size: 0.9rem;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ===================== INITIALIZE ===================== #
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize VectorStore in session state
@st.cache_resource
def get_vectordb():
    return VectorStore(dim=384)

vectordb = get_vectordb()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Initialize current Q&A (the active question-answer pair)
if "current_qa" not in st.session_state:
    st.session_state.current_qa = None

# ===================== HEADER ===================== #
st.markdown('<h1 class="main-title">📄 RAG Document Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Powered by Gemini 2.5 Flash + FAISS Hybrid Search</p>', unsafe_allow_html=True)

# ===================== SIDEBAR ===================== #
with st.sidebar:
    st.markdown("## 🚀 Upload Documents")
    st.markdown("---")
    
    # Multi-file uploader
    uploaded_files = st.file_uploader(
        "Drop your files here",
        type=["txt", "pdf", "docx", "pptx", "png", "jpg", "jpeg", "webp"],
        help="Supported formats: TXT, PDF, DOCX, PPTX, Images (OCR)",
        accept_multiple_files=True  # Enable multiple file selection
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} file(s) selected")
        
        if st.button("✨ Upload & Index All", use_container_width=True):
            # Save all files first
            file_paths = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                file_paths.append((file_path, uploaded_file.name))
            
            # Process files in parallel
            progress_bar = st.progress(0)
            status_container = st.empty()
            results_container = st.container()
            
            successful = []
            failed = []
            
            def process_single_file(file_info):
                """Process a single file and return result"""
                file_path, file_name = file_info
                try:
                    result = build_index(file_path, vectordb)
                    return {"status": "success", "name": file_name, "result": result}
                except Exception as e:
                    return {"status": "error", "name": file_name, "error": str(e)}
            
            # Process all files simultaneously (cap at 10 workers for safety)
            max_workers = min(len(file_paths), 10)
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_single_file, fp): fp for fp in file_paths}
                completed = 0
                
                for future in as_completed(futures):
                    completed += 1
                    progress_bar.progress(completed / len(file_paths))
                    
                    result = future.result()
                    if result["status"] == "success":
                        successful.append(result)
                        st.session_state.uploaded_files.append(result["name"])
                    else:
                        failed.append(result)
                    
                    status_container.text(f"⏳ Processed {completed}/{len(file_paths)} files...")
            
            # Show final results
            status_container.empty()
            progress_bar.empty()
            
            with results_container:
                if successful:
                    st.success(f"✅ Successfully indexed {len(successful)} document(s)!")
                    with st.expander("📋 View Details", expanded=False):
                        for s in successful:
                            st.markdown(f"**{s['name']}**: {s['result']['chunks_added']} chunks")
                
                if failed:
                    st.error(f"❌ Failed to index {len(failed)} document(s)")
                    with st.expander("⚠️ View Errors", expanded=True):
                        for f in failed:
                            st.markdown(f"**{f['name']}**: {f['error']}")
    
    st.markdown("---")
    
    # Reset Database Button
    if st.button("⚠️ Reset Database", use_container_width=True, type="secondary"):
        if "confirm_reset" not in st.session_state:
            st.session_state.confirm_reset = True
            st.warning("⚠️ Click again to confirm. This will delete ALL indexed documents!")
            st.rerun()
        else:
            # Delete all uploaded files
            import shutil
            if os.path.exists(UPLOAD_DIR):
                shutil.rmtree(UPLOAD_DIR)
                os.makedirs(UPLOAD_DIR, exist_ok=True)
            
            # Reset vector database
            vectordb.reset()
            
            # Clear session state
            st.session_state.chat_history = []
            st.session_state.uploaded_files = []
            st.session_state.current_qa = None
            st.session_state.confirm_reset = False
            
            st.success("✅ Database reset successfully!")
            st.rerun()

    
    st.markdown("---")
    st.markdown("### 🛠️ Powered By")
    st.markdown("""
    - 🤖 **Gemini 2.5 Flash**
    - 🦙 **Groq Llama 3.1** (Fallback)
    - 🔍 **FAISS + BM25** Hybrid Search
    - 📦 **BGE Small** Embeddings
    """)

# ===================== MAIN CHAT AREA ===================== #
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 💬 Ask Your Question")
    
with col2:
    pass

# Question Input
question = st.text_input(
    "Type your question here...",
    placeholder="What would you like to know from your documents?",
    label_visibility="collapsed"
)

# ===================== CURRENT ANSWER DISPLAY (Below Question Box) ===================== #
if st.session_state.current_qa:
    # current_qa now contains a list of answers
    answers_list = st.session_state.current_qa if isinstance(st.session_state.current_qa, list) else [st.session_state.current_qa]
    
    for idx, chat in enumerate(answers_list):
        with st.container():
            # Show question if multiple questions
            if len(answers_list) > 1:
                st.markdown(f'<div class="question-bubble">❓ {chat["question"]}</div>', unsafe_allow_html=True)
            
            # Answer bubble
            answer_text = chat.get("summary") or chat.get("answer_raw") or "⚠️ No answer generated."
            st.markdown(f'<div class="answer-bubble">{answer_text}</div>', unsafe_allow_html=True)
            
            # Show raw answer if different from summary
            if chat.get("answer_raw") and chat.get("summary") != chat.get("answer_raw") and chat.get("summary"):
                with st.expander("📄 View Full Answer"):
                    st.write(chat["answer_raw"])
            
            # Display points if available
            if chat.get("points"):
                st.markdown("**✅ Key Points:**")
                for point in chat["points"]:
                    st.markdown(f"- {point}")
            
            # Display sources if available
            if chat.get("sources"):
                st.markdown("**📚 Sources:**")
                sources_html = " ".join([f'<span class="source-tag">{src}</span>' for src in chat["sources"]])
                st.markdown(sources_html, unsafe_allow_html=True)
            
            # Add spacing between multiple answers
            if idx < len(answers_list) - 1:
                st.markdown("<br>", unsafe_allow_html=True)

# Ask Button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    ask_button = st.button("🚀 Ask Question", use_container_width=True)

if ask_button:
    if not question.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        # Move current Q&A to chat history before asking new question
        if st.session_state.current_qa is not None:
            # If current_qa is a list, add each answer individually to history
            if isinstance(st.session_state.current_qa, list):
                st.session_state.chat_history.extend(st.session_state.current_qa)
            else:
                st.session_state.chat_history.append(st.session_state.current_qa)
        
        with st.spinner("🧠 Thinking..."):
            try:
                result = answer_multiple_questions(question, vectordb)
                
                # Store ALL answers as current Q&A (not in history yet)
                if result.get("answers"):
                    # Store all answers as a list
                    st.session_state.current_qa = [
                        {
                            "question": ans.get("question", question),
                            "summary": ans.get("summary", ""),
                            "answer_raw": ans.get("answer_raw", ""),
                            "points": ans.get("points", []),
                            "sources": ans.get("sources", []),
                        }
                        for ans in result["answers"]
                    ]
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ===================== CHAT HISTORY DISPLAY ===================== #
st.markdown("---")

if st.session_state.chat_history:
    st.markdown("### 💬 Chat History")
    
    # Display chat history in reverse order (newest first)
    for i, chat in enumerate(reversed(st.session_state.chat_history), start=1):
        with st.container():
            # Question bubble
            st.markdown(f'<div class="question-bubble">❓ {chat["question"]}</div>', unsafe_allow_html=True)
            
            # Answer bubble
            answer_text = chat.get("summary") or chat.get("answer_raw") or "⚠️ No answer generated."
            st.markdown(f'<div class="answer-bubble">{answer_text}</div>', unsafe_allow_html=True)
            
            # Show raw answer if different from summary
            if chat.get("answer_raw") and chat.get("summary") != chat.get("answer_raw") and chat.get("summary"):
                with st.expander("📄 View Full Answer"):
                    st.write(chat["answer_raw"])
            
            # Display points if available
            if chat.get("points"):
                st.markdown("**✅ Key Points:**")
                for point in chat["points"]:
                    st.markdown(f"- {point}")
            
            # Display sources if available
            if chat.get("sources"):
                st.markdown("**📚 Sources:**")
                sources_html = " ".join([f'<span class="source-tag">{src}</span>' for src in chat["sources"]])
                st.markdown(sources_html, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
else:
    st.info("💡 No questions asked yet. Upload documents and start asking questions!")

# ===================== FOOTER ===================== #
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #64748b; font-size: 0.85rem;">
    <p>Built with ❤️ using Streamlit & Gemini 2.5</p>
</div>
""", unsafe_allow_html=True)
