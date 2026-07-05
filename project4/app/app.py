from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st
import sys

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bert_model import predict_bert
from src.emoji_features import extract_emoji_sentiment
from src.feature_engineering import transform_features
from src.model import load_object, predict_sentiment
from src.preprocessing import clean_text


@st.cache_resource
def load_resources():
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "sentiment_model.joblib"
    vectorizer_path = project_root / "models" / "vectorizer.joblib"

    model = load_object(str(model_path))
    vectorizer = load_object(str(vectorizer_path))
    return model, vectorizer


def predict_ml(input_text: str, model, vectorizer):
    cleaned_text = clean_text(input_text)
    if not cleaned_text:
        return "Neutral"

    data = pd.DataFrame(
        [{"sentence": input_text, "clean_sentence": cleaned_text}]
    )
    features = transform_features(data, vectorizer)
    prediction = predict_sentiment(model, features)[0]
    return "Positive" if int(prediction) == 1 else "Negative"


def main():
    st.set_page_config(page_title="Sentiment Analysis", page_icon="�", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        :root {
            --primary-color: #6366f1;
            --secondary-color: #ec4899;
            --success-color: #10b981;
            --danger-color: #ef4444;
            --neutral-color: #f3f4f6;
        }
        
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        .stTitle {
            font-size: 3em;
            font-weight: 900;
            background: linear-gradient(45deg, #6366f1, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5em;
        }
        
        .stSubheader {
            color: #475569;
            font-size: 1.3em;
            font-weight: 600;
        }
        
        .sentiment-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-left: 5px solid;
            margin: 10px 0;
        }
        
        .sentiment-positive {
            border-left-color: #10b981;
        }
        
        .sentiment-negative {
            border-left-color: #ef4444;
        }
        
        .sentiment-neutral {
            border-left-color: #f59e0b;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header with gradient
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center; color: #6366f1; font-size: 2.5em; font-weight: 900;'>💬 Sentiment Analysis</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #64748b; font-size: 1.1em; font-weight: 500;'>Powered by ML + BERT + Emoji Detection</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Description
    st.markdown("""
    <div style='background: #f0f9ff; border-left: 4px solid #0284c7; padding: 15px; border-radius: 8px; margin-bottom: 20px;'>
        <p style='color: #0c4a6e; font-size: 1em; font-weight: 500;'>
        🤖 Enter a comment and choose a prediction model to analyze the sentiment instantly!
        </p>
    </div>
    """, unsafe_allow_html=True)

    user_input = st.text_area("✏️ Enter your comment", "I love this product! 😊", height=120)

    try:
        model, vectorizer = load_resources()
    except FileNotFoundError as error:
        st.error(f"❌ Could not load model resources: {error}")
        return
    except Exception as error:
        st.error(f"❌ Unexpected error when loading resources: {error}")
        return

    # Emoji sentiment section
    st.markdown("### 😊 Emoji Sentiment Analysis")
    emoji_data = extract_emoji_sentiment(user_input)
    
    emoji_col1, emoji_col2, emoji_col3, emoji_col4 = st.columns(4)
    with emoji_col1:
        st.metric("😊 Positive", emoji_data['emoji_positive'], delta=None)
    with emoji_col2:
        st.metric("😢 Negative", emoji_data['emoji_negative'], delta=None)
    with emoji_col3:
        st.metric("🎯 Total", emoji_data['emoji_total'], delta=None)
    with emoji_col4:
        score_value = emoji_data['emoji_score']
        st.metric("📊 Score", score_value, delta=None)
    
    st.markdown("---")
    
    # Prediction buttons section
    st.markdown("### 🔮 Choose Prediction Model")
    button_col1, button_col2 = st.columns(2)
    
    with button_col1:
        if st.button("🤖 ML Model Prediction", use_container_width=True, key="ml_button"):
            with st.spinner("⏳ Analyzing with ML model..."):
                sentiment = predict_ml(user_input, model, vectorizer)
                sentiment_emoji = "✅" if sentiment == "Positive" else "❌"
                color = "#10b981" if sentiment == "Positive" else "#ef4444"
                st.markdown(f"""
                <div class='sentiment-card sentiment-{'positive' if sentiment == 'Positive' else 'negative'}'>
                    <p style='color: {color}; font-size: 1.5em; font-weight: 700;'>{sentiment_emoji} {sentiment}</p>
                    <p style='color: #64748b; font-size: 0.9em;'>ML Model (TF-IDF + Logistic Regression)</p>
                </div>
                """, unsafe_allow_html=True)
    
    with button_col2:
        if st.button("🧠 BERT Model Prediction", use_container_width=True, key="bert_button"):
            with st.spinner("⏳ Analyzing with BERT..."):
                bert_sentiment, confidence = predict_bert(user_input)
                sentiment_emoji = "✅" if bert_sentiment == "Positive" else ("❌" if bert_sentiment == "Negative" else "❓")
                color_map = {"Positive": "#10b981", "Negative": "#ef4444", "Neutral": "#f59e0b"}
                color = color_map.get(bert_sentiment, "#64748b")
                st.markdown(f"""
                <div class='sentiment-card sentiment-{bert_sentiment.lower()}'>
                    <p style='color: {color}; font-size: 1.5em; font-weight: 700;'>{sentiment_emoji} {bert_sentiment}</p>
                    <p style='color: #64748b; font-size: 0.9em;'>BERT Model (Multilingual Transformer)</p>
                    <p style='color: {color}; font-size: 1.1em; font-weight: 600; margin-top: 8px;'>Confidence: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(45deg, #6366f1, #ec4899); padding: 20px; border-radius: 10px; text-align: center; margin-top: 30px;'>
        <p style='color: white; font-size: 0.95em; font-weight: 500; margin: 0;'>
        ⚡ Built with ML (TF-IDF + Logistic Regression) + BERT + Emoji Detection
        </p>
        <p style='color: #e0e7ff; font-size: 0.85em; margin: 5px 0 0 0;'>
        Powered by Streamlit • Transformers • scikit-learn
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
