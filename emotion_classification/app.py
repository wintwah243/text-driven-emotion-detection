import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB

# --- Page configuration ---
st.set_page_config(
    page_title="Text-driven Emotion Detection Analysis Project with Inside out cartoon characters (IS-211)",
    layout="centered"
)

# --- Sidebar navigation ---
menu = ["Home", "Classifier", "Members"]
choice = st.sidebar.selectbox("Navigation", menu)

# --- Load and train the model once ---
@st.cache_resource
def train_model():
    df = pd.read_excel("mainfile_dataCleaning.xlsx")  
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label_text'])

    # Drop classes with <2 samples
    label_counts = df['label_encoded'].value_counts()
    valid_labels = label_counts[label_counts >= 2].index
    df = df[df['label_encoded'].isin(valid_labels)]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=10000)
    X = vectorizer.fit_transform(df['text'])
    y = df['label_encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = ComplementNB()
    model.fit(X_train, y_train)

    return model, vectorizer, le

model, vectorizer, le = train_model()

# --- inside out cartoon images ---
emotion_images = {
    "joy": "https://i.pinimg.com/1200x/da/74/e3/da74e3594f00c4498270a80d61474c11.jpg",
    "sadness": "https://i.pinimg.com/736x/29/69/1f/29691f8adb89f742309fcc84cf9641ce.jpg",
    "anger": "https://i.pinimg.com/736x/86/39/e0/8639e06333cd4b295e1f0a904d125029.jpg",
    "fear": "https://i.pinimg.com/1200x/39/6e/4d/396e4d4774d0dbc95146d5940be32ea9.jpg",
    "suprise": "https://i.pinimg.com/1200x/0e/33/a5/0e33a5ff2bd24858db5db50da575f703.jpg",
    "love": "https://i.pinimg.com/1200x/04/c7/54/04c7544d34b87376bb0fb84d2ba29ec4.jpg"
}

# --- Home / Intro Page ---
if choice == "Home":
    st.title("Welcome to Our Emotion Classifier Project!")
    st.write("""
    This project classifies emotions from your text and shows a matching **Inside Out cartoon character**.
    Type some text and see which character expresses your feelings!
    """)
    st.image("https://i.pinimg.com/1200x/55/e3/8f/55e38f8e45cb57d7b41678d44664abfd.jpg", width=300)
    st.write("Use the sidebar to go to the **Classifier** page.")

# --- Classifier Page ---
elif choice == "Classifier":
    st.title("Text Driven Emotion Classifier")
    st.subheader("Enter a text to classify its emotion")
    user_input = st.text_area("Type your text here:")

    if st.button("Classify Emotion"):
        if user_input.strip():
            X_input = vectorizer.transform([user_input])
            pred = model.predict(X_input)
            emotion = le.inverse_transform(pred)[0]

            st.success(f"Predicted Emotion: **{emotion}**")
            if emotion.lower() in emotion_images:
                st.image(emotion_images[emotion.lower()], caption=f"{emotion} Character", width=250)
            else:
                st.info("No cartoon available for this emotion.")
        else:
            st.warning("Please enter some text before classifying.")

# --- Members Page ---
elif choice == "Members":
    st.title("Project Members")
    st.write("Meet the team behind this project:")
    st.markdown("""
    - **Wint Wah Kyaw Soe** – Streamlit App Integration   
    - **Ingyin Phyu** – Knowledge Representation  
    - **May Me Wai Zin** – Model Train  
    - **Aye Yu Mon** – Data Preprocessing & Cleaning  
    - **Kalayar Lin Mon** – Power BI Visualization
    - **Shwe YaMin** – Power BI Visualization  
    - **Khin Chaw Thant** – Applied ETL Process  
    - **Phoo Pwint Han** – Applied ETL Process    
    """)
