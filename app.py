import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

from preprocessing import clean_text

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS FOR STYLING
# ==========================================
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Result Cards */
    .result-card-positive {
        background: linear-gradient(135deg, #1e3c27 0%, #2e5c3c 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-top: 20px;
    }
    .result-card-negative {
        background: linear-gradient(135deg, #4a191f 0%, #70262f 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-top: 20px;
    }
    .result-card-neutral {
        background: linear-gradient(135deg, #2a2a2a 0%, #3a3a3a 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-top: 20px;
    }
    .result-card-irrelevant {
        background: linear-gradient(135deg, #2e4a62 0%, #3e6382 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-top: 20px;
    }
    
    /* Emotion text styling */
    .emotion-text {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .confidence-text {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Header typography */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CONSTANTS & CACHING
# ==========================================
MODEL_PATH = "model/lstm_model.h5"
TOKENIZER_PATH = "model/tokenizer.pkl"
LABEL_ENCODER_PATH = "model/label_encoder.pkl"
DATA_PATH = "data/twitter_training.csv"
MAX_LEN = 40 # Based on training notebook

@st.cache_resource(show_spinner="Loading ML Model...")
def load_ml_components():
    """Load the trained model, tokenizer, and label encoder."""
    try:
        model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open(LABEL_ENCODER_PATH, 'rb') as handle:
            le = pickle.load(handle)
        return model, tokenizer, le, True
    except Exception as e:
        return None, None, None, False

@st.cache_data(show_spinner="Loading Dataset...")
def load_dataset():
    """Load the original dataset for insights."""
    try:
        df = pd.read_csv(DATA_PATH, header=None, names=["id", "entity", "sentiment", "text"])
        # Same cleaning applied during training for consistency in stats
        df = df.dropna(subset=["text"]).reset_index(drop=True)
        df = df.drop_duplicates(subset=["text", "sentiment"]).reset_index(drop=True)
        return df
    except Exception as e:
        return None

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def predict_sentiment(text, model, tokenizer, le):
    """Predicts sentiment for a given text."""
    # 1. Clean Text
    cleaned_text = clean_text(text)
    
    if len(cleaned_text) == 0:
        return None, 0.0
        
    # 2. Tokenize & Pad
    seq = tokenizer.texts_to_sequences([cleaned_text])
    pad_seq = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    
    # 3. Predict
    prediction = model.predict(pad_seq)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    
    # 4. Decode Label
    predicted_label = le.inverse_transform([predicted_class_idx])[0]
    
    return predicted_label, confidence

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🐦 Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Analyze Sentiment", "Dataset Insights", "About Model"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Status")

# Try to load models
model, tokenizer, le, models_loaded = load_ml_components()

if models_loaded:
    st.sidebar.success("✅ Model Loaded Successfully")
    st.sidebar.markdown(f"**Classes:** {', '.join(le.classes_)}")
else:
    st.sidebar.error("❌ Model Files Missing")
    st.sidebar.caption("Please place lstm_model.h5, tokenizer.pkl, and label_encoder.pkl in the model/ directory.")

# ==========================================
# MAIN CONTENT
# ==========================================

if page == "Home":
    st.title("Twitter Sentiment Analysis")
    st.markdown("### Discover the hidden emotions behind every tweet.")
    
    st.write("""
    Welcome to the Twitter Sentiment Analysis Dashboard! This application uses advanced Deep Learning (LSTM) 
    to automatically determine the sentiment of social media posts.
    
    👈 **Use the sidebar** to navigate through the application:
    - **Analyze Sentiment:** Paste a tweet and get instant predictions.
    - **Dataset Insights:** Explore the data that powered this AI.
    - **About Model:** Learn about the neural network architecture.
    """)
    
    st.image("https://images.unsplash.com/photo-1611605698335-8b1569810432?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80", 
             use_container_width=True, 
             caption="AI-Powered Social Media Analysis")
    
    st.info("💡 Pro Tip: Navigate to 'Analyze Sentiment' to test the model with your own text.")

elif page == "Analyze Sentiment":
    st.title("🔍 Analyze Tweet Sentiment")
    st.write("Enter a tweet below to analyze its sentiment using our trained AI model.")
    
    if not models_loaded:
        st.error("Model files are missing. Cannot perform predictions. Please check the README instructions.")
    else:
        user_input = st.text_area("Tweet Text", placeholder="E.g., I absolutely love the new borderlands game, it's amazing!", height=150)
        
        # Character counter
        char_count = len(user_input)
        st.caption(f"{char_count} characters")
        
        if st.button("Analyze Sentiment", type="primary", use_container_width=True):
            if not user_input.strip():
                st.warning("Please enter some text to analyze.")
            else:
                with st.spinner("Analyzing text patterns..."):
                    label, confidence = predict_sentiment(user_input, model, tokenizer, le)
                    
                    if label is None:
                        st.warning("The text provided doesn't contain enough valid characters to analyze after cleaning (e.g., only emojis or punctuation).")
                    else:
                        st.markdown("### Result")
                        
                        # Determine card style and emoji based on label
                        if label == "Positive":
                            card_class = "result-card-positive"
                            emoji = "✨😊"
                        elif label == "Negative":
                            card_class = "result-card-negative"
                            emoji = "😠📉"
                        elif label == "Neutral":
                            card_class = "result-card-neutral"
                            emoji = "😐📊"
                        else: # Irrelevant
                            card_class = "result-card-irrelevant"
                            emoji = "🤷‍♂️🚫"
                            
                        # Display custom result card
                        st.markdown(f"""
                        <div class="{card_class}">
                            <div class="emotion-text">{emoji} {label}</div>
                            <div class="confidence-text">AI Confidence Score: {confidence*100:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show preprocessed text for transparency
                        with st.expander("See how the AI sees your text (Preprocessing)"):
                            st.code(clean_text(user_input))

elif page == "Dataset Insights":
    st.title("📊 Dataset Insights")
    st.write("Explore the training data that taught the AI how to understand emotions.")
    
    df = load_dataset()
    
    if df is not None:
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Tweets", f"{len(df):,}")
        col2.metric("Sentiment Classes", len(df['sentiment'].unique()))
        col3.metric("Entities Monitored", len(df['entity'].dropna().unique()))
        
        st.markdown("---")
        
        # Charts
        st.subheader("Sentiment Distribution")
        
        # Prepare data for chart
        sentiment_counts = df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        # Color mapping matching our UI
        color_discrete_map = {
            'Positive': '#2e5c3c',
            'Negative': '#70262f',
            'Neutral': '#3a3a3a',
            'Irrelevant': '#3e6382'
        }
        
        fig = px.bar(
            sentiment_counts, 
            x='Sentiment', 
            y='Count',
            color='Sentiment',
            color_discrete_map=color_discrete_map,
            text='Count',
            template="plotly_dark"
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Data Preview
        st.subheader("Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True)
        
    else:
        st.error("Dataset not found. Please ensure 'twitter_training.csv' is located in the data/ directory.")

elif page == "About Model":
    st.title("🧠 About The Model")
    st.write("Technical details regarding the Machine Learning architecture behind this application.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Architecture
        This application utilizes a **Long Short-Term Memory (LSTM)** neural network, which is a specialized type of Recurrent Neural Network (RNN) perfectly suited for sequence data like text.
        
        **Model Pipeline:**
        1. **Input:** Raw text (Max 40 tokens).
        2. **Embedding Layer:** Converts words into dense vectors of fixed size (128 dimensions), capturing semantic meanings.
        3. **Spatial Dropout 1D (0.3):** Prevents overfitting by dropping entire 1D feature maps instead of individual elements.
        4. **Bidirectional LSTM (128 units):** Reads the sequence forwards and backwards, gathering context from both past and future words.
        5. **LSTM (64 units):** Further processes the sequence to extract higher-level features.
        6. **Dense Layers:** Fully connected layers (64 units -> 4 units) with ReLU and Softmax activations to output final class probabilities.
        
        ### Text Preprocessing
        Before feeding text to the model, it undergoes rigorous cleaning:
        - Lowercasing all characters
        - Removing URLs and mentions (`@user`)
        - Removing `#` from hashtags but keeping the word itself
        - Stripping out punctuation (keeping only letters and apostrophes)
        - Tokenization (Vocabulary Size: 20,000 words)
        - Sequence Padding (Max Length: 40)
        """)
        
    with col2:
        st.info("""
        ### Quick Specs
        - **Vocabulary Size:** 20,000 words
        - **Max Sequence Length:** 40 tokens
        - **Embedding Dimension:** 128
        - **Optimizer:** Adam
        - **Loss Function:** Categorical Crossentropy
        """)
