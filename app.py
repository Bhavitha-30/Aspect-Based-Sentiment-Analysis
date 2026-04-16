import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import pickle
import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag  
from nltk import ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources
nltk_resources = ['stopwords', 'wordnet', 'averaged_perceptron_tagger', 'punkt', 'maxent_ne_chunker', 'words']
for resource in nltk_resources:
    try:
        nltk.data.find(f'corpora/{resource}' if resource != 'averaged_perceptron_tagger' else f'taggers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Aspect keywords mapping (domain knowledge)
ASPECT_KEYWORDS = {
    'quality': ['quality', 'durability', 'build', 'material', 'construction', 'solid', 'strong', 'weak', 'broken'],
    'performance': ['performance', 'speed', 'charging', 'charge', 'fast', 'slow', 'responsive', 'efficient', 'lag', 'smooth'],
    'design': ['design', 'appearance', 'look', 'color', 'aesthetic', 'style', 'elegant', 'ugly', 'beautiful'],
    'price': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'value', 'money', 'worth', 'overpriced'],
    'durability': ['durable', 'lasting', 'longevity', 'lifespan', 'reliable', 'breakable', 'fragile'],
    'delivery': ['delivery', 'shipping', 'packaging', 'arrived', 'prompt', 'delayed', 'damage', 'condition'],
    'functionality': ['feature', 'features', 'function', 'functions', 'interface', 'ui', 'ux', 'work', 'operate', 'operates', 'use', 'usable', 'practical', 'useful', 'experience'],
    'size': ['size', 'compact', 'big', 'small', 'large', 'dimension', 'space', 'fit'],
    'weight': ['weight', 'heavy', 'light', 'portable'],
    'comfort': ['comfort', 'comfortable', 'ergonomic', 'painful', 'cozy', 'awkward'],
    'camera': ['camera', 'photo', 'picture', 'lens', 'zoom', 'focus', 'selfie', 'image'],
    'battery': ['battery', 'battery life', 'drain', 'drains', 'discharge', 'backup'],
}

# Sentiment lexicons
# positive lexicon count: 32
# negative lexicon count: 29
# neutral lexicon count: 10
POSITIVE_WORDS = set([
    'excellent', 'amazing', 'fantastic', 'great', 'awesome', 'wonderful', 'perfect',
    'love', 'best', 'brilliant', 'outstanding', 'superb', 'good', 'nice', 'beautiful',
    'highly recommended', 'worth', 'happy', 'satisfied', 'impressed', 'impressive', 'cool',
    'premium', 'stylish', 'smooth', 'fast', 'responsive', 'efficient', 'elegant',
    'reliable', 'comfortable', 'ergonomic', 'cozy', 'durable', 'lasting', 'longevity',
    'practical', 'useful', 'affordable', 'prompt', 'solid', 'strong', 'excellent value'
])

NEGATIVE_WORDS = set([
    'bad', 'poor', 'terrible', 'awful', 'horrible', 'useless', 'waste', 'disappointing',
    'hate', 'worst', 'broken', 'defective', 'cheap', 'low quality', 'unhappy', 'unsatisfied',
    'regret', 'disappointed', 'pathetic', 'disgusting', 'annoying',
    'slow', 'expensive', 'overpriced', 'weak', 'breakable', 'fragile', 'ugly',
    'painful', 'awkward', 'heavy', 'lag', 'delayed', 'damage', 'poorly', 'issue', 'issues',
    'glitch', 'bugs', 'buggy', 'frustrating', 'unusable', 'difficult', 'worse'
])

NEUTRAL_WORDS = set([
    'average', 'acceptable', 'okay', 'ok', 'decent', 'fair', 'moderate', 'standard',
    'normal', 'typical', 'adequate', 'fine', 'middle', 'meets expectations'
])

def preprocess_text(text):
    """Clean and normalize text"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_aspects(text):
    """Extract product aspects from review text using keyword matching and linguistic patterns"""
    cleaned_text = preprocess_text(text)
    aspects_found = set()
    
    # Keyword-based extraction
    for aspect, keywords in ASPECT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in cleaned_text.lower():
                aspects_found.add(aspect)
                break
    
    # If no aspects found, use dependency parsing heuristic
    if not aspects_found:
        words = cleaned_text.split()
        common_aspects = ['product', 'item', 'stuff', 'thing']
        for word in words:
            if word in common_aspects:
                aspects_found.add('overall')
    
    return list(aspects_found) if aspects_found else ['overall']

def split_sentences(text):
    """Split text into sentences with a fallback if NLTK tokenizer is unavailable."""
    try:
        return sent_tokenize(text)
    except Exception:
        cleaned = text.strip()
        if not cleaned:
            return []
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        return [sent for sent in sentences if sent]


def split_clauses(text):
    """Split a sentence into smaller clauses so aspect mentions can be isolated."""
    cleaned = preprocess_text(text)
    clauses = re.split(r'\bbut\b|\bhowever\b|\balthough\b|\byet\b|;|\.|\!|\?', cleaned)
    return [clause.strip() for clause in clauses if clause.strip()]


def lexicon_sentiment(text):
    """Fallback lexicon-based sentiment detection for short aspect clauses."""
    cleaned = preprocess_text(text)
    positive_score = sum(1 for phrase in POSITIVE_WORDS if phrase in cleaned)
    negative_score = sum(1 for phrase in NEGATIVE_WORDS if phrase in cleaned)
    neutral_score = sum(1 for phrase in NEUTRAL_WORDS if phrase in cleaned)

    if neutral_score >= positive_score and neutral_score >= negative_score and neutral_score > 0:
        return 'neutral'
    if positive_score > negative_score and positive_score >= neutral_score:
        return 'positive'
    if negative_score > positive_score and negative_score >= neutral_score:
        return 'negative'
    return None


def extract_aspect_clause(text, aspect_keywords):
    """Extract the clause or clause group most closely related to the aspect."""
    sentences = split_sentences(text)
    matched_clauses = []
    for sentence in sentences:
        cleaned_sentence = preprocess_text(sentence)
        if any(keyword in cleaned_sentence for keyword in aspect_keywords):
            clauses = split_clauses(sentence)
            for clause in clauses:
                if any(keyword in clause for keyword in aspect_keywords):
                    matched_clauses.append(clause)
    if matched_clauses:
        return ' '.join(matched_clauses)

    for sentence in sentences:
        cleaned_sentence = preprocess_text(sentence)
        if any(keyword in cleaned_sentence for keyword in aspect_keywords):
            return cleaned_sentence

    return preprocess_text(text)


def calculate_aspect_sentiment(text, aspect, vectorizer_model):
    """Calculate sentiment specifically for an aspect in the text."""
    vectorizer, model = vectorizer_model
    aspect_keywords = get_aspect_keywords_in_text(text, aspect)
    target_text = extract_aspect_clause(text, aspect_keywords) if aspect_keywords else preprocess_text(text)

    lexicon_prediction = lexicon_sentiment(target_text)
    if lexicon_prediction is not None:
        return lexicon_prediction, 1.0

    try:
        vectorized = vectorizer.transform([target_text])
        prediction = model.predict(vectorized)[0]
        confidence = max(model.predict_proba(vectorized)[0])
    except Exception:
        prediction = 'neutral'
        confidence = 0.5

    return prediction, confidence


def get_aspect_keywords_in_text(text, aspect):
    """Find keywords related to an aspect in the text"""
    keywords = ASPECT_KEYWORDS.get(aspect, [])
    cleaned_text = preprocess_text(text).lower()
    found_keywords = [kw for kw in keywords if kw in cleaned_text]
    return found_keywords

@st.cache_resource
def load_or_train_model():
    """Load or train the sentiment classification model"""
    with st.spinner("🔄 Loading and training model... This may take a minute."):
        # Load dataset
        df = pd.read_csv("Dataset-SA.csv")
        
        # Data cleaning
        df = df.dropna(subset=['Review', 'Sentiment'])
        df['Review'] = df['Review'].astype(str)
        df['Sentiment'] = df['Sentiment'].str.lower()
        
        # Display dataset info
        st.sidebar.info(f"**Dataset Summary:**\n"
                       f"• Total Reviews: {len(df):,}\n"
                       f"• Unique Products: {df['product_name'].nunique()}\n"
                       f"• Avg Review Length: {df['Review'].str.len().mean():.0f} chars")
        
        st.sidebar.info(f"**Sentiment Distribution:**\n{df['Sentiment'].value_counts().to_string()}")
        
        # Preprocess
        df['cleaned_text'] = df['Review'].apply(preprocess_text)
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
        X = vectorizer.fit_transform(df['cleaned_text'])
        y = df['Sentiment']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        # Train model
        model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred, labels=['negative', 'neutral', 'positive']),
            'labels': ['negative', 'neutral', 'positive'],
            'classification_report': classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        }
        
        return vectorizer, model, metrics

def sentiment_to_rating(sentiment):
    """Convert sentiment to a star rating for display"""
    sentiment_map = {
        'positive': 5,
        'neutral': 3,
        'negative': 1
    }
    return sentiment_map.get(sentiment, 3)


def analyze_review(review, vectorizer_model):
    """Extract aspects and calculate sentiment details for a single review."""
    aspects = extract_aspects(review)
    aspect_details = []

    for aspect in aspects:
        sentiment, confidence = calculate_aspect_sentiment(review, aspect, vectorizer_model)
        aspect_details.append({
            'aspect': aspect,
            'sentiment': sentiment,
            'confidence': confidence,
            'keywords': get_aspect_keywords_in_text(review, aspect)
        })

    report = generate_summary_report([(detail['aspect'], detail['sentiment']) for detail in aspect_details])
    return aspects, aspect_details, report


def format_aspects_summary(aspect_details):
    if not aspect_details:
        return "No aspects detected"
    return "; ".join([f"{detail['aspect'].title()}: {detail['sentiment'].title()}" for detail in aspect_details])


def generate_summary_report(aspects_sentiments):
    positive_count = sum(1 for item in aspects_sentiments if item[1] == 'positive')
    negative_count = sum(1 for item in aspects_sentiments if item[1] == 'negative')
    neutral_count = sum(1 for item in aspects_sentiments if item[1] == 'neutral')
    total_count = positive_count + negative_count + neutral_count

    if positive_count > negative_count:
        overall_sentiment = 'positive'
    elif negative_count > positive_count:
        overall_sentiment = 'negative'
    else:
        overall_sentiment = 'neutral'

    report = {
        'total_aspects': total_count,
        'positive_aspects': positive_count,
        'negative_aspects': negative_count,
        'neutral_aspects': neutral_count,
        'overall_sentiment': overall_sentiment,
    }
    return report


def generate_simple_summary(aspect_details):
    summary = []
    
    for detail in aspect_details:
        stars = sentiment_to_rating(detail['sentiment'])
        star_display = '⭐' * stars
        summary.append(f"{detail['aspect'].title()} {star_display}")
    
    return "<br>".join(summary)

# ==================== STREAMLIT UI ====================

st.set_page_config(
    page_title="Aspect-Based Sentiment Analysis",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .positive-aspect {
        background-color: #d4edda;
        padding: 10px;
        border-left: 4px solid #28a745;
        margin: 5px 0;
        border-radius: 4px;
    }
    .negative-aspect {
        background-color: #f8d7da;
        padding: 10px;
        border-left: 4px solid #dc3545;
        margin: 5px 0;
        border-radius: 4px;
    }
    .neutral-aspect {
        background-color: #e2e3e5;
        padding: 10px;
        border-left: 4px solid #6c757d;
        margin: 5px 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("⭐ Aspect-Based Sentiment Analysis (ABSA)")
st.markdown("""
### Detailed Product Review Analysis
Extract fine-grained sentiment insights about specific product aspects from customer reviews.

**How it works:** Enter a product review, and the system will identify key aspects (quality, price, design, etc.) 
and classify the sentiment for each aspect separately.
""")

# Load model
vectorizer, model, metrics = load_or_train_model()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("📊 Model Performance")
    
    col1, col2 = st.columns(2)
    col1.metric("🎯 Accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("⭐ F1-Score", f"{metrics['f1']:.2%}")
    
    col1, col2 = st.columns(2)
    col1.metric("🎪 Precision", f"{metrics['precision']:.2%}")
    col2.metric("🔍 Recall", f"{metrics['recall']:.2%}")
    
    with st.expander("📈 Confusion Matrix"):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='RdYlGn', 
                   xticklabels=metrics['labels'], yticklabels=metrics['labels'], ax=ax)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        st.pyplot(fig)
        plt.close()
    
    st.divider()
    st.subheader("📋 Available Aspects")
    aspects_list = list(ASPECT_KEYWORDS.keys())
    for i, aspect in enumerate(aspects_list, 1):
        st.caption(f"{i}. **{aspect.title()}** - {len(ASPECT_KEYWORDS[aspect])} keywords")

# ==================== MAIN CONTENT ====================

# Tabs for different features
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analyze Review", "📊 Batch Analysis", "📚 Examples", "📖 Guide"])

# ========== TAB 1: SINGLE REVIEW ANALYSIS ==========
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Product Review")
        user_input = st.text_area(
            "Paste or type a customer review:",
            height=150,
            placeholder="E.g., The camera quality is excellent but the battery backup is poor...",
            key="user_review"
        )
    
    with col2:
        st.subheader("ℹ️ About ABSA")
        st.markdown("""
        **Aspect-Based Sentiment Analysis:**
        
        - Extracts **product aspects**
        - Classifies **sentiment per aspect**
        - Shows **confidence scores**
        - Identifies **relevant keywords**
        
        **Detected Aspects:**
        - Quality, Performance
        - Design, Price
        - Durability, Delivery
        - And more...
        """)
    
    # Analysis button
    if st.button("🔍 Analyze Review", use_container_width=True, key="analyze_btn"):
        if not user_input or not user_input.strip():
            st.error("❌ Please enter a review text.")
        elif len(user_input) < 10:
            st.error("❌ Review must be at least 10 characters long.")
        elif len(user_input) > 1000:
            st.error("❌ Review cannot exceed 1000 characters.")
        else:
            st.divider()
            
            aspects, aspect_details, report = analyze_review(user_input, (vectorizer, model))
            
            # Display results
            st.subheader("📋 Analysis Results")

            # Simple summary output
            simple_summary = generate_simple_summary(aspect_details)
            st.subheader("📝 Summary Output")
            st.markdown(simple_summary, unsafe_allow_html=True)
            
            # Summary metrics
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("🎯 Total Aspects", report['total_aspects'])
            with metric_col2:
                st.metric("✅ Positive", report['positive_aspects'])
            with metric_col3:
                st.metric("❌ Negative", report['negative_aspects'])
            with metric_col4:
                st.metric("⚪ Neutral", report['neutral_aspects'])
            
            # Overall sentiment
            overall_sentiment = report['overall_sentiment'].upper()
            overall_color = '🟢' if overall_sentiment == 'POSITIVE' else ('🔴' if overall_sentiment == 'NEGATIVE' else '⚪')
            st.markdown(f"### Overall Sentiment: {overall_color} **{overall_sentiment}**")
            
            # Aspect-level analysis
            st.subheader("🔍 Aspect-Level Breakdown")
            
            col1, col2 = st.columns([1.5, 2])
            
            with col1:
                st.markdown("#### Aspect Summary")
                for detail in sorted(aspect_details, key=lambda x: x['confidence'], reverse=True):
                    aspect_title = detail['aspect'].title()
                    sentiment = detail['sentiment'].upper()
                    confidence = detail['confidence']
                    
                    # Color based on sentiment
                    if detail['sentiment'] == 'positive':
                        sentiment_emoji = '🟢'
                        color = '#d4edda'
                    elif detail['sentiment'] == 'negative':
                        sentiment_emoji = '🔴'
                        color = '#f8d7da'
                    else:
                        sentiment_emoji = '⚪'
                        color = '#e2e3e5'
                    
                    # Star rating
                    stars = sentiment_to_rating(detail['sentiment'])
                    star_display = '⭐' * stars + '☆' * (5 - stars)
                    
                    st.markdown(f"""
                    <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 8px 0;">
                    <b>{aspect_title}</b> {sentiment_emoji}<br>
                    <small>{star_display}<br>
                    Confidence: {confidence:.1%}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### Keywords Found")
                for detail in aspect_details:
                    if detail['keywords']:
                        st.markdown(f"**{detail['aspect'].title()}:**")
                        st.write(", ".join([f"`{kw}`" for kw in detail['keywords']]))
                    else:
                        st.caption(f"*{detail['aspect'].title()}: No specific keywords detected*")
            
            # Star rating visualization
            st.subheader("⭐ Aspect Ratings")
            
            # Create a sorted list of aspects by their aspect name
            sorted_details = sorted(aspect_details, key=lambda x: x['aspect'])
            
            rating_data = []
            for detail in sorted_details:
                stars = sentiment_to_rating(detail['sentiment'])
                rating_data.append({
                    'Aspect': detail['aspect'].title(),
                    'Stars': stars,
                    'Sentiment': detail['sentiment'].title()
                })
            
            # Display as horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 4))
            aspects_names = [d['Aspect'] for d in rating_data]
            stars = [d['Stars'] for d in rating_data]
            colors = ['#28a745' if s >= 4 else ('#ffc107' if s >= 3 else '#dc3545') for s in stars]
            
            bars = ax.barh(aspects_names, stars, color=colors)
            ax.set_xlim(0, 5)
            ax.set_xlabel('Rating (Stars)', fontsize=12, fontweight='bold')
            ax.set_title('Aspect-Based Sentiment Ratings', fontsize=14, fontweight='bold')
            
            # Add star labels on bars
            for i, (bar, star_count) in enumerate(zip(bars, stars)):
                star_text = '⭐' * int(star_count) + '☆' * (5 - int(star_count))
                ax.text(star_count + 0.1, i, star_text, va='center', fontsize=10)
            
            st.pyplot(fig)
            plt.close()
            
            # Detailed insights
            st.subheader("💡 Key Insights")
            
            insights = []
            if report['positive_aspects'] > 0:
                insights.append(f"✅ **{report['positive_aspects']} positive aspect(s)** found - Customer liked these features")
            if report['negative_aspects'] > 0:
                insights.append(f"⚠️ **{report['negative_aspects']} negative aspect(s)** found - Areas for improvement")
            if report['neutral_aspects'] > 0:
                insights.append(f"⚪ **{report['neutral_aspects']} neutral aspect(s)** - Mixed or unclear opinions")
            
            for insight in insights:
                st.markdown(f"- {insight}")
            
            # Review metadata
            st.subheader("📊 Review Metadata")
            review_col1, review_col2, review_col3 = st.columns(3)
            review_col1.metric("📝 Text Length", f"{len(user_input)} chars")
            review_col2.metric("📌 Word Count", f"{len(user_input.split())} words")
            review_col3.metric("🎯 Aspects Detected", len(aspects))

# ========== TAB 2: BATCH ANALYSIS ==========
with tab2:
    st.subheader("📊 Batch Review Analysis")
    st.write("Upload or paste multiple reviews for batch analysis and comparison.")
    
    batch_input_method = st.radio("Choose input method:", ["📝 Paste Reviews", "📤 Upload CSV"])
    
    batch_reviews = []
    
    if batch_input_method == "📝 Paste Reviews":
        batch_text = st.text_area(
            "Enter reviews (one per line):",
            height=200,
            placeholder="Review 1\nReview 2\nReview 3"
        )
        if batch_text:
            batch_reviews = [r.strip() for r in batch_text.split('\n') if r.strip()]
    else:
        uploaded_file = st.file_uploader("Upload CSV with reviews", type=['csv'])
        if uploaded_file:
            try:
                batch_df = pd.read_csv(uploaded_file)
                review_columns = [col for col in batch_df.columns if 'review' in col.lower() or 'text' in col.lower() or 'comment' in col.lower()]

                if review_columns:
                    selected_col = st.selectbox("Select review column", review_columns, index=0)
                    batch_reviews = batch_df[selected_col].astype(str).tolist()
                elif batch_df.shape[1] == 1:
                    batch_reviews = batch_df.iloc[:, 0].astype(str).tolist()
                else:
                    st.warning("No obvious review/text column found. Using the first column by default.")
                    batch_reviews = batch_df.iloc[:, 0].astype(str).tolist()

                st.markdown("**Preview uploaded data:**")
                st.dataframe(batch_df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    if st.button("📊 Analyze Batch", use_container_width=True) and batch_reviews:
        if len(batch_reviews) > 100:
            st.warning("⚠️ Limiting to first 100 reviews for performance")
            batch_reviews = batch_reviews[:100]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        batch_results = []
        
        for idx, review in enumerate(batch_reviews):
            if len(review.strip()) > 10:
                _, aspect_details, report = analyze_review(review, (vectorizer, model))
                batch_results.append({
                    'Review': review[:100] + '...' if len(review) > 100 else review,
                    'Aspects': len(aspect_details),
                    'Aspect Summary': format_aspects_summary(aspect_details),
                    'Positive': report['positive_aspects'],
                    'Negative': report['negative_aspects'],
                    'Neutral': report['neutral_aspects'],
                    'Overall': report['overall_sentiment'].title()
                })
            
            progress = (idx + 1) / len(batch_reviews)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {idx + 1}/{len(batch_reviews)}")
        
        progress_bar.empty()
        status_text.empty()
        
        # Display results table
        results_df = pd.DataFrame(batch_results)
        st.dataframe(results_df, use_container_width=True)
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        summary_col1.metric("Total Reviews Analyzed", len(results_df))
        summary_col2.metric("Avg Aspects per Review", f"{results_df['Aspects'].mean():.1f}")
        summary_col3.metric("Positive Reviews", (results_df['Overall'] == 'Positive').sum())
        summary_col4.metric("Negative Reviews", (results_df['Overall'] == 'Negative').sum())
        
        # Sentiment distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Pie chart
        sentiment_counts = results_df['Overall'].value_counts()
        colors_pie = {'Positive': '#28a745', 'Negative': '#dc3545', 'Neutral': '#6c757d'}
        colors_list = [colors_pie.get(s, '#6c757d') for s in sentiment_counts.index]
        axes[0].pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors_list, startangle=90)
        axes[0].set_title('Overall Sentiment Distribution')
        
        # Bar chart for aspect counts
        aspect_counts = results_df[['Positive', 'Negative', 'Neutral']].sum()
        axes[1].bar(['Positive', 'Negative', 'Neutral'], aspect_counts.values, color=['#28a745', '#dc3545', '#6c757d'])
        axes[1].set_title('Total Aspect Sentiments')
        axes[1].set_ylabel('Count')
        
        st.pyplot(fig)
        plt.close()
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="absa_batch_results.csv",
            mime="text/csv"
        )

# ========== TAB 3: EXAMPLES ==========
with tab3:
    st.subheader("📚 Try Example Reviews")
    st.write("Click an example to instantly see how ABSA analysis works.")
    
    examples = {
        "Positive Review": "The camera quality is excellent and the battery backup is fantastic. The design is sleek and modern. Highly recommended!",
        "Negative Review": "The product quality is poor and it broke after a week. The delivery was delayed and packaging was damaged. Very disappointed.",
        "Mixed Review": "The performance is amazing but the price is quite expensive. Design is beautiful though.",
        "Detailed Positive": "Amazing product! The build quality is solid, fast performance, looks elegant, great value for money, and delivered on time.",
        "Detailed Negative": "Terrible purchase. Low quality material, slow performance, ugly design, overpriced compared to competitors, and poor customer service.",
        "Delivery Focus": "Product arrived quickly and was well packaged. However, the quality is average and it's not as durable as expected.",
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        for label in list(examples.keys())[:2]:
            if st.button(f"📖 {label}", use_container_width=True, key=f"example_{label}"):
                st.session_state.example_selected = label
    
    with col2:
        for label in list(examples.keys())[2:4]:
            if st.button(f"📖 {label}", use_container_width=True, key=f"example_{label}"):
                st.session_state.example_selected = label
    
    with col3:
        for label in list(examples.keys())[4:]:
            if st.button(f"📖 {label}", use_container_width=True, key=f"example_{label}"):
                st.session_state.example_selected = label
    
    # Display selected example
    if 'example_selected' in st.session_state:
        example_text = examples[st.session_state.example_selected]
        st.markdown(f"### 📖 {st.session_state.example_selected}")
        st.info(f"*\"{example_text}\"*")
        
        # Analyze it
        st.markdown("**Analysis:**")
        
        _, aspect_details, report = analyze_review(example_text, (vectorizer, model))
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Aspects", report['total_aspects'])
        col2.metric("Positive", report['positive_aspects'])
        col3.metric("Negative", report['negative_aspects'])
        
        overall_sentiment = report['overall_sentiment'].upper()
        overall_emoji = '🟢' if overall_sentiment == 'POSITIVE' else ('🔴' if overall_sentiment == 'NEGATIVE' else '⚪')
        st.markdown(f"**Overall Sentiment:** {overall_emoji} **{overall_sentiment}**")
        
        st.write("**Aspect Breakdown:**")
        for detail in sorted(aspect_details, key=lambda x: x['confidence'], reverse=True):
            stars = sentiment_to_rating(detail['sentiment'])
            star_display = '⭐' * stars + '☆' * (5 - stars)
            st.write(f"- **{detail['aspect'].title()}**: {star_display} ({detail['sentiment'].title()}, {detail['confidence']:.1%})")

# ========== TAB 4: GUIDE ==========
with tab4:
    st.subheader("📖 Understanding ABSA")
    
    st.markdown("""
    ## What is Aspect-Based Sentiment Analysis (ABSA)?
    
    Aspect-Based Sentiment Analysis (ABSA) is a fine-grained approach to analyzing customer opinions 
    by identifying specific product aspects and determining the sentiment expressed toward each aspect separately.
    
    ### Traditional vs. Aspect-Based Sentiment Analysis
    
    **Traditional Sentiment Analysis:**
    - Reviews only one overall sentiment
    - Example: "The camera is great but battery is poor" → Overall: Positive/Negative
    
    **Aspect-Based Sentiment Analysis:**
    - Analyzes sentiment for each specific aspect
    - Example: "The camera is great but battery is poor" → Camera: Positive, Battery: Negative
    
    ### Key Features of This System
    
    1. **Aspect Extraction**
       - Automatically identifies product aspects mentioned in reviews
       - Aspects: Quality, Performance, Design, Price, Durability, Delivery, Functionality, Size, Weight, Comfort
    
    2. **Sentiment Classification**
       - Classifies sentiment for each aspect: Positive, Negative, or Neutral
       - Uses TF-IDF vectorization + Logistic Regression
    
    3. **Confidence Scores**
       - Shows how confident the model is about each prediction
       - Helps identify uncertain classifications
    
    4. **Keyword Extraction**
       - Identifies which specific keywords triggered aspect detection
       - Provides transparency in the analysis
    
    5. **Visual Ratings**
       - Star ratings (1-5) for each aspect
       - Easy-to-understand visualization
    
    ### Business Applications
    
    - 📈 **Product Improvement**: Identify which aspects need improvement
    - 🎯 **Targeted Marketing**: Highlight strengths in marketing campaigns
    - 💬 **Customer Insights**: Understand detailed customer feedback
    - 🏆 **Competitive Analysis**: Compare aspects with competitors
    - 🔍 **Quality Control**: Monitor specific feature performance over time
    
    ### Example Use Case
    
    **Customer Review:**
    > "The build quality is excellent and the design is sleek. However, the price is too high and battery life is disappointing."
    
    **ABSA Output:**
    - Quality: ⭐⭐⭐⭐⭐ (Positive)
    - Design: ⭐⭐⭐⭐⭐ (Positive)
    - Price: ⭐⭐ (Negative)
    - Battery: ⭐⭐ (Negative)
    
    **Business Insights:**
    1. Product excels in quality and design
    2. Pricing strategy needs review
    3. Battery performance is a concern
    4. Overall mixed sentiment - focus on price and battery
    
    ### Model Performance
    
    This model is trained on 200,000+ product reviews with:
    - Accuracy: See sidebar metrics
    - Multi-class classification (Positive, Negative, Neutral)
    - TF-IDF vectorization with bigrams
    - Balanced class weights for fair classification
    
    ### Tips for Best Results
    
    ✅ **Do:**
    - Use complete sentences with clear language
    - Include specific product aspects in reviews
    - Provide detailed feedback
    
    ❌ **Avoid:**
    - Single-word reviews
    - Ambiguous or unclear language
    - Reviews longer than 1000 characters
    - Non-English text
    
    ### Aspect Categories Detected
    """)
    
    # Create aspect explanation table
    aspect_info = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        aspect_info.append({
            'Aspect': aspect.title(),
            'Examples': ', '.join(keywords[:3]) + '...',
            'Count': len(keywords)
        })
    
    aspect_df = pd.DataFrame(aspect_info)
    st.dataframe(aspect_df, use_container_width=True)
    
    st.markdown("""
    ### Sentiment Classes
    
    | Sentiment | Description | Rating |
    |-----------|-------------|--------|
    | **Positive** | Customer satisfied, praising the aspect | ⭐⭐⭐⭐⭐ |
    | **Neutral** | Mixed opinions, factual statements | ⭐⭐⭐ |
    | **Negative** | Customer dissatisfied, criticizing the aspect | ⭐⭐ |
    
    """)

# Footer
st.divider()
st.caption("🔬 Built with TF-IDF + Logistic Regression | Data: ~200K Labeled Product Reviews | " +
           "Model: Aspect-Based Sentiment Analysis (ABSA)")