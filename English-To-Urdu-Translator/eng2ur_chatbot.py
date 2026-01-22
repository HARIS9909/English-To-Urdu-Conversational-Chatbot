import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import time
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="English to Urdu Translator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
        margin: 1rem 0;
    }
    .urdu-text {
        font-size: 1.5rem;
        color: #059669;
        direction: rtl;
        text-align: right;
        font-family: 'Noto Nastaliq Urdu', 'Segoe UI', sans-serif;
        padding: 1rem;
        background-color: #F0F9FF;
        border-radius: 0.5rem;
        border: 1px solid #E5E7EB;
    }
    .stButton button {
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        border: none;
    }
    .stButton button:hover {
        background-color: #1D4ED8;
    }
    .model-info {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #F59E0B;
    }
</style>
""", unsafe_allow_html=True)

# Title and Header
st.markdown('<h1 class="main-header">🤖 English to Urdu Translator</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Trained with dedication using fine-tuned models • Your personal translation assistant</p>', unsafe_allow_html=True)

# Initialize session state for chat history
if 'history' not in st.session_state:
    st.session_state.history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'translations_count' not in st.session_state:
    st.session_state.translations_count = 0

@st.cache_resource(show_spinner="Loading the translation model...")
def load_model():
    """Load the translation model with progress indication"""
    names = ["finetuned-opus", "Helsinki-NLP/opus-mt-en-ur"]
    tokenizer = None
    model = None
    loaded_name = None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, name in enumerate(names):
        status_text.text(f"Attempting to load: {name}")
        progress_bar.progress((idx + 1) * 50)
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSeq2SeqLM.from_pretrained(name)
            loaded_name = name
            status_text.text(f"✅ Successfully loaded: {name}")
            time.sleep(0.5)
            break
        except Exception as e:
            status_text.text(f"❌ Could not load {name}, trying next...")
            time.sleep(0.5)
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    if model is not None:
        device = "cpu"
        model.to(device)
        st.session_state.model_loaded = True
        return tokenizer, model, device, loaded_name
    else:
        st.error("❌ Could not load any model. Please check your model files.")
        return None, None, None, None

# Sidebar
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    
    # Model loading section
    if st.button("🚀 Load Translation Model", type="primary"):
        with st.spinner("Loading model... This might take a moment"):
            tokenizer, model, device, loaded_name = load_model()
            if model:
                st.success(f"✅ Model loaded successfully: {loaded_name}")
                st.session_state.tokenizer = tokenizer
                st.session_state.model = model
                st.session_state.device = device
                st.session_state.loaded_name = loaded_name
    
    st.markdown("---")
    
    # Quick examples
    # st.markdown("### 💡 Quick Examples")
    examples = [
        # "hello",
        # "i want to talk",
        # "my friend lives in Karachi",
        # "can you help me",
        # "pakistan won the match",
        # "thank you very much",
        # "where is the bus stop?"
    ]
    
    for example in examples:
        if st.button(f"• {example}", key=f"ex_{example}"):
            st.session_state.example_text = example
    
    st.markdown("---")
    
    # Statistics
    st.markdown("### 📊 Statistics")
    st.metric("Translations Today", st.session_state.translations_count)
    st.metric("Chat History", len(st.session_state.history))
    
    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.session_state.translations_count = 0
        st.success("History cleared!")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### ✍️ Input English Text")
    
    # Text input with example
    if 'example_text' in st.session_state:
        default_text = st.session_state.example_text
        del st.session_state.example_text
    else:
        default_text = ""
    
    input_text = st.text_area(
        "Enter English text to translate:",
        value=default_text,
        height=150,
        placeholder="Type your English text here...",
        key="input_text"
    )
    
    # Translation button
    col1_1, col1_2, col1_3 = st.columns([1, 1, 1])
    with col1_2:
        translate_clicked = st.button("Translate 🔄", type="primary", use_container_width=True)

with col2:
    st.markdown("### 📝 Urdu Translation")
    
    if translate_clicked and input_text.strip():
        if not st.session_state.model_loaded:
            st.warning("⚠️ Please load the model first from the sidebar!")
        else:
            with st.spinner("Translating..."):
                try:
                    # Prepare inputs
                    inputs = st.session_state.tokenizer(
                        input_text, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True
                    )
                    inputs = {k: v.to(st.session_state.device) for k, v in inputs.items()}
                    
                    # Generate translation
                    with torch.no_grad():
                        output_ids = st.session_state.model.generate(
                            **inputs,
                            max_new_tokens=64,
                            num_beams=5,
                            length_penalty=1.1,
                            no_repeat_ngram_size=3,
                            early_stopping=True,
                        )
                    
                    urdu_translation = st.session_state.tokenizer.decode(
                        output_ids[0], 
                        skip_special_tokens=True
                    )
                    
                    # Display translation
                    st.markdown(f'<div class="urdu-text">{urdu_translation}</div>', unsafe_allow_html=True)
                    
                    # Add to history
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state.history.append({
                        'time': timestamp,
                        'english': input_text,
                        'urdu': urdu_translation
                    })
                    
                    # Update counter
                    st.session_state.translations_count += 1
                    
                    # Success message
                    st.success("✅ Translation successful!")
                    
                    # Copy button
                    st.code(urdu_translation, language=None)
                    
                except Exception as e:
                    st.error(f"❌ Translation error: {str(e)}")
    else:
        st.info("👈 Enter text and click 'Translate' to see Urdu translation here")

# History Section
st.markdown("---")
st.markdown("### 📜 Translation History")

if st.session_state.history:
    # Convert history to DataFrame for display
    history_df = pd.DataFrame(st.session_state.history)
    
    # Display in reverse chronological order
    for idx, item in enumerate(reversed(st.session_state.history)):
        with st.expander(f"#{len(st.session_state.history)-idx} | {item['time']}"):
            st.markdown(f"**English:** {item['english']}")
            st.markdown(f"**Urdu:** {item['urdu']}")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("📋 Copy Urdu", key=f"copy_{idx}"):
                    st.code(item['urdu'])
                    st.toast("Copied to clipboard!", icon="✅")
            
            with col_btn2:
                if st.button("🗑️ Delete", key=f"del_{idx}"):
                    # Find and remove the item
                    for i, hist_item in enumerate(st.session_state.history):
                        if hist_item['time'] == item['time'] and hist_item['english'] == item['english']:
                            del st.session_state.history[i]
                            st.rerun()
                            break
else:
    st.info("No translations yet. Start by translating some text!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #6B7280; padding: 1rem;">
        <p>Built with ❤️ using Streamlit • Fine-tuned Translation Model • Designed for simplicity</p>
        <p>Made possible by your hard work in training the model! 🎉</p>
    </div>
    """, 
    unsafe_allow_html=True
)