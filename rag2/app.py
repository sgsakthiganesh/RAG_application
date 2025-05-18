import streamlit as st
import requests

# ---------- Streamlit UI ----------
st.title("World History from 1900 to present")
query = st.text_input("Answers the questions about the world events from 1900", placeholder="Ask your doubts here...")
st.markdown("###  Question")

# ---------- Call FastAPI Backend ----------
FASTAPI_URL = "http://localhost:8000/ask"  

if query:
    with st.spinner("Loading..."):
        try:
            response = requests.post(FASTAPI_URL, json={"question": query})
            if response.status_code == 200:
                answer = response.json()["answer"]
                st.markdown("###  Answer")
                st.success(answer)
            else:
                st.error(f"Error from backend: {response.text}")
        except Exception as e:
            st.error(f"Could not connect to backend: {e}")
