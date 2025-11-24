import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import shap
import lime
import lime.lime_text
import streamlit.components.v1 as components

# ============================================================================
# Page Configuration & Title
# ============================================================================
st.set_page_config(page_title="Fake News Analysis", layout="wide")
st.title("Advanced Fake News Detection & Analysis")

# ============================================================================
# Data Loading and Model Training (Cached)
# ============================================================================
@st.cache_data(show_spinner="Loading data and training model...")
def load_data_and_train_model():
    """
    Loads data, trains Passive-Aggressive Classifier, and prepares SHAP explainer.
    """
    # Try loading with the specific path provided, fall back to relative path
    try:
        df = pd.read_csv(r"C:\Users\tanma\Documents\project\news.csv")
    except FileNotFoundError:
        try:
            df = pd.read_csv("news.csv")
        except FileNotFoundError:
            st.error("Error: 'news.csv' not found. Please place it in the project directory.")
            st.stop()
        
    df.dropna(inplace=True)
    
    # Robust column handling
    try:
        labels = df['label']
    except KeyError:
        st.error(f"Error: Column 'label' not found. Available columns: {list(df.columns)}")
        st.stop()
    
    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=42)
    
    # Initialize and Fit Vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
    tfidf_test = tfidf_vectorizer.transform(x_test)
    
    # Initialize and Fit Model
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)
    
    # Calculate Accuracy
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    
    # Prepare text for WordClouds
    real_news_text = ' '.join(df[df['label'] == 'REAL']['text'])
    fake_news_text = ' '.join(df[df['label'] == 'FAKE']['text'])
    
    return pac, tfidf_vectorizer, score, x_train, real_news_text, fake_news_text

# Load Data & Model
model, vectorizer, accuracy, X_train_text, real_news_text, fake_news_text = load_data_and_train_model()

# ============================================================================
# Initialize SHAP Explainer (Cached separately to save time)
# ============================================================================
@st.cache_resource(show_spinner="Initializing SHAP explainer...")
def get_shap_explainer(_model, _vectorizer, _train_text):
    # We take a small sample (100) of the training data as the "background" for SHAP
    # Using the full dataset would be too memory intensive
    background_sample = _train_text.sample(100, random_state=42)
    background_vec = _vectorizer.transform(background_sample)
    
    # LinearExplainer is best for PassiveAggressiveClassifier
    explainer = shap.LinearExplainer(_model, background_vec)
    return explainer

shap_explainer = get_shap_explainer(model, vectorizer, X_train_text)

# ============================================================================
# Sidebar Navigation
# ============================================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to", 
    ("Project Overview", "Live Fake News Detector", "Keyword Analysis", "Model Explainability")
)

# ============================================================================
# PAGE 1: Project Overview
# ============================================================================
if page == "Project Overview":
    st.header("Understanding the Fight Against Misinformation")
    st.markdown("""
    This project demonstrates a complete pipeline for detecting and analyzing fake news using Natural Language Processing (NLP) and Machine Learning.
    
    ### Key Features:
    1.  **Live Detection:** Classify any news article text as REAL or FAKE in real-time.
    2.  **Keyword Analysis:** Visualize indicative words using Word Clouds.
    3.  **Model Explainability:** Uses **LIME** and **SHAP** to visualize decision-making.
    """)
    st.subheader("Model Performance")
    st.info(f"The **Passive-Aggressive Classifier** model has an accuracy of **{accuracy*100:.2f}%**.")

# ============================================================================
# PAGE 2: Live Fake News Detector
# ============================================================================
elif page == "Live Fake News Detector":
    st.header("Classify a News Article")
    news_text = st.text_area("News Article Text", height=250, placeholder="Paste article content here...")
    
    if st.button("Analyze News"):
        if news_text:
            with st.spinner('Analyzing...'):
                vectorized_text = vectorizer.transform([news_text])
                prediction = model.predict(vectorized_text)[0]
                
                # Since PAC doesn't have predict_proba, we use decision_function
                decision_val = model.decision_function(vectorized_text)[0]
                confidence = 1 / (1 + np.exp(-abs(decision_val))) # Sigmoid-like confidence
                
                if prediction == 'FAKE':
                    st.error(f"Prediction: FAKE (Confidence: {confidence:.2%})")
                else:
                    st.success(f"Prediction: REAL (Confidence: {confidence:.2%})")
        else:
            st.warning("Please enter some text.")

# ============================================================================
# PAGE 3: Keyword Analysis
# ============================================================================
elif page == "Keyword Analysis":
    st.header("Word Cloud Visualization")
    st.write("Most frequent words in Fake vs Real news.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    wordcloud_fake = WordCloud(width=800, height=500, background_color='black', colormap='Reds').generate(fake_news_text)
    ax1.imshow(wordcloud_fake, interpolation='bilinear')
    ax1.set_title('FAKE News Keywords', fontsize=20)
    ax1.axis('off')

    wordcloud_real = WordCloud(width=800, height=500, background_color='white', colormap='Greens').generate(real_news_text)
    ax2.imshow(wordcloud_real, interpolation='bilinear')
    ax2.set_title('REAL News Keywords', fontsize=20)
    ax2.axis('off')

    st.pyplot(fig)

# ============================================================================
# PAGE 4: Model Explainability (LIME & SHAP)
# ============================================================================
elif page == "Model Explainability":
    st.header("Why did the model make this prediction?")
    st.markdown("We use **LIME** and **SHAP** to open the 'black box' of the AI model.")

    sample_text = st.selectbox(
        "Choose a sample article or enter your own:",
        (
            "Donald Trump Sends Out Bizarre New Year’s Eve Message; This is Disturbing", 
            "House Republicans Fret About Winning Their Health Care Suit",
            "Enter my own text..."
        )
    )

    if sample_text == "Enter my own text...":
        user_input = st.text_area("Enter text to explain:", height=150)
        text_to_explain = user_input
    else:
        text_to_explain = sample_text

    if st.button("Generate Explanations"):
        if not text_to_explain:
            st.warning("Please provide text to explain.")
        else:
            st.info("Generating explanations... This may take a few seconds.")
            
            # 1. LIME Explanation
            st.markdown("### 1. LIME Explanation")
            st.write("LIME highlights specific words in the text that contributed to the prediction.")
            
            def predict_proba_wrapper(texts):
                vec = vectorizer.transform(texts)
                scores = model.decision_function(vec)
                probs_fake = 1 / (1 + np.exp(-scores))
                probs_real = 1 - probs_fake
                return np.vstack((probs_fake, probs_real)).T

            with st.spinner("Running LIME..."):
                lime_explainer = lime.lime_text.LimeTextExplainer(class_names=['FAKE', 'REAL'])
                exp = lime_explainer.explain_instance(
                    text_to_explain, predict_proba_wrapper, num_features=10
                )
                components.html(exp.as_html(), height=400, scrolling=True)

            # 2. SHAP Explanation
            st.markdown("---")
            st.markdown("### 2. SHAP Explanation")
            st.write("SHAP shows the magnitude and direction (positive/negative) of each word's impact.")

            with st.spinner("Running SHAP..."):
                # Transform input
                vec_input = vectorizer.transform([text_to_explain])
                
                # Calculate SHAP values
                shap_values = shap_explainer.shap_values(vec_input)
                
                # Feature names
                feature_names = vectorizer.get_feature_names_out()
                
                # --- FIX START: Correctly capturing Matplotlib figures ---
                
                # FORCE PLOT
                st.write("**SHAP Force Plot:** Visualizes how features push the prediction from the base value.")
                try:
                    # Generate the plot with matplotlib=True and show=False
                    shap.plots.force(
                        shap_explainer.expected_value, 
                        shap_values[0], 
                        feature_names=feature_names, 
                        matplotlib=True,
                        show=False
                    )
                    # Grab the current figure (gcf) that SHAP just drew
                    fig = plt.gcf()
                    # Render it in Streamlit
                    st.pyplot(fig, bbox_inches='tight')
                    # Clear the figure to prevent overlapping with the next plot
                    plt.clf()
                except Exception as e:
                    st.error(f"Could not generate Force Plot: {e}")

                # BAR PLOT
                st.write("**SHAP Bar Plot:** Top words influencing this specific prediction.")
                try:
                    # Create Explanation object
                    shap_exp = shap.Explanation(
                        values=shap_values[0],
                        base_values=shap_explainer.expected_value,
                        data=vec_input.toarray()[0], 
                        feature_names=feature_names
                    )
                    
                    # Draw bar plot
                    shap.plots.bar(shap_exp, max_display=10, show=False)
                    
                    # Grab the figure and render
                    fig = plt.gcf()
                    st.pyplot(fig, bbox_inches='tight')
                    plt.clf()
                except Exception as e:
                    st.error(f"Could not generate Bar Plot: {e}")
                # --- FIX END ---