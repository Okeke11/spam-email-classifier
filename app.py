import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# --- Page Configuration ---
st.set_page_config(page_title="Spam Detector AI", layout="wide")

st.title("📧 SMS Spam Classifier AI")
st.markdown("""
This app uses **Natural Language Processing (NLP)** to detect if a message is **Spam** or **Ham** (Legitimate). 
It is trained on the UCI SMS Spam Collection Dataset (5,574 real messages).
""")

# --- 1. Load Real Data (Cached) ---
# We use @st.cache_data so we don't reload/retrain every time the user clicks a button.
@st.cache_data
def load_data_and_train_model():
    # URL to the raw dataset (Tab-separated values)
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    
    # Load data
    df = pd.read_table(url, header=None, names=['label', 'message'])
    
    # Convert labels to numbers (ham=0, spam=1) for easier processing if needed, 
    # though Naive Bayes handles strings fine, numbers are often safer.
    df['label_num'] = df.label.map({'ham': 0, 'spam': 1})
    
    # Vectorization
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df.message)
    y = df.label
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    
    # Calculate Accuracy
    acc = accuracy_score(y_test, clf.predict(X_test))
    
    return clf, vectorizer, df, acc

# Load everything
clf, vectorizer, df, accuracy = load_data_and_train_model()

# Display Model Performance
st.sidebar.header("Model Performance")
st.sidebar.write(f"Training Data Size: {len(df)} messages")
st.sidebar.write(f"Model Accuracy: **{accuracy * 100:.2f}%**")

# --- 2. Visualization Section ---
st.write("---")
st.header("📊 Data Visualization: What does Spam look like?")

if st.checkbox("Show Word Clouds"):
    st.write("Generating visualizations... (this handles 5,000+ messages!)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Words in SPAM 🚨")
        # Join all spam messages into one giant text block
        spam_text = " ".join(df[df['label']=='spam']['message'])
        # Generate cloud
        spam_cloud = WordCloud(width=400, height=300, background_color='black', colormap='Reds').generate(spam_text)
        # Display
        fig, ax = plt.subplots()
        ax.imshow(spam_cloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        
    with col2:
        st.subheader("Top Words in HAM (Real) ✅")
        ham_text = " ".join(df[df['label']=='ham']['message'])
        ham_cloud = WordCloud(width=400, height=300, background_color='white', colormap='Greens').generate(ham_text)
        fig, ax = plt.subplots()
        ax.imshow(ham_cloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

# --- 3. The Prediction Frontend ---
st.write("---")
st.header("🤖 Live Detector")

user_input = st.text_area("Enter a message to test:", height=100, placeholder="e.g., Congratulations! You've won a $1000 Walmart gift card. Click here to claim.")

if st.button("Analyze Message"):
    if user_input:
        # 1. Vectorize the user input
        vect_input = vectorizer.transform([user_input])
        # 2. Predict
        prediction = clf.predict(vect_input)[0]
        # 3. Get probabilities (confidence)
        probs = clf.predict_proba(vect_input)[0] # Returns [prob_ham, prob_spam]
        
        # Display Result
        if prediction == 'spam':
            st.error(f"🚨 SPAM DETECTED (Confidence: {probs[1]*100:.1f}%)")
        else:
            st.success(f"✅ THIS SEEMS SAFE (Confidence: {probs[0]*100:.1f}%)")
    else:
        st.warning("Please enter some text first.")