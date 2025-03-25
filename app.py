import streamlit as st
import pandas as pd
import numpy as np
import pickle
import spacy

# Load model and data
model = pickle.load(open("model.pkl", "rb"))
symptoms = list(pd.read_csv("Training.csv").columns[:-1])
nlp = spacy.load("en_core_web_sm")

# Intent keywords for basic rule-based detection
intent_keywords = {
    "greeting": ["hello", "hi", "hey"],
    "symptom_check": ["fever", "cough", "symptom", "pain", "headache", "cold", "nausea", "sick"],
    "appointment": ["book", "appointment", "schedule", "visit", "meet"],
    "thanks": ["thanks", "thank", "bye"]
}

# Function to detect intent
def detect_intent(text):
    doc = nlp(text.lower())
    for token in doc:
        for intent, keywords in intent_keywords.items():
            if token.lemma_ in keywords:
                return intent
    return "unknown"

# Streamlit app starts here
#st.title("🩺 AI HealthBot - Enhanced Chatbot")
#st.write("Chat with a simple AI health assistant to check symptoms, book appointments, or ask general health questions.")

# Streamlit UI
st.set_page_config(page_title="AI HealthBot - Enhanced Chatbot", page_icon="🩺")
st.title("🩺 AI HealthBot - Enhanced Chatbot")
st.write("Talk to me about your symptoms, appointments, or general health questions!")


# Session state to remember previous intent
if "last_intent" not in st.session_state:
    st.session_state.last_intent = None

# User input
user_input = st.text_input("You:")

if user_input:
    intent = detect_intent(user_input)
    st.session_state.last_intent = intent

    if intent == "greeting":
        st.write("👋 Hello! How can I assist you today?")
    elif intent == "symptom_check":
        st.write("🩺 Please select your symptoms below:")
        selected_symptoms = st.multiselect("Symptoms", symptoms)
        if st.button("Predict Disease"):
            input_data = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]
            prediction = model.predict([input_data])[0]
            st.success(f"🔍 Predicted Disease: **{prediction}**")
            st.warning("⚠️ Please consult a doctor for an accurate diagnosis.")
    elif intent == "appointment":
        st.write("📅 You can book an appointment by calling **123-456-7890** or visiting our [website](#).")
    elif intent == "thanks":
        st.write("🙏 You're welcome! Stay healthy and take care.")
    else:
        st.write("🤖 I'm still learning. Try asking about symptoms, appointments, or greetings.")
