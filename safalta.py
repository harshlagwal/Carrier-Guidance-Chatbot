import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# ✅ Streamlit Page Setup
st.set_page_config(page_title="Safalta Apki - Career Guidance", layout="wide")

# ✅ Custom Styling (Black Background, Blue Title)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000; /* 🔹 Black Background */
        color: white;
    }
    [data-testid="stSidebar"] {
        background-color: #1e1e1e !important;
        color: white;
    }
    .stTitle {
        color: #007BFF !important; /* 🔹 Blue Title */
        font-size: 32px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Model & Tokenizer Loading (Efficient with Caching)
@st.cache_resource
def load_model():
    model_name = "MBZUAI/LaMini-T5-738M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# ✅ Function to Get AI Response
def get_chatbot_response(user_input):
    try:
        input_text = f"Provide career advice for: {user_input}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=300, repetition_penalty=1.8, temperature=0.9, top_k=40, top_p=0.85)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(". ", ".\n\n")
    except Exception as e:
        return f"⚠ Error: {str(e)}"

# ✅ Sidebar - User Details Form
st.sidebar.header("👤 Fill Your Details")
name = st.sidebar.text_input("📝 Your Name")
age = st.sidebar.number_input("🎂 Your Age", min_value=10, max_value=60, value=18)
education = st.sidebar.selectbox("🎓 Your Education Level", ["High School", "Undergraduate", "Postgraduate", "Diploma", "Other"])
interests = st.sidebar.text_area("🎯 Your Career Interests", "e.g. Software Development, Data Science, Business Management")
skills = st.sidebar.text_area("🛠 Your Skills", "e.g. Python, Communication, Leadership")
experience = st.sidebar.selectbox("💼 Work Experience", ["Yes", "No"])

if st.sidebar.button("💾 Save Information"):
    st.sidebar.success("✅ User Information Saved Successfully!")

if name:
    st.sidebar.markdown("### 👤 User Details")
    st.sidebar.markdown(f"**Name:** {name}  \n**Age:** {age}  \n**Education:** {education}  \n**Interests:** {interests}  \n**Skills:** {skills}  \n**Experience:** {experience}")

# ✅ Main Chatbot Section
st.markdown('<h1 class="stTitle">🚀 Safalta Apki - Career Guidance Chatbot</h1>', unsafe_allow_html=True)
st.markdown("💡 **Ask any career-related question and get instant advice!**")

user_query = st.text_input("🔍 Ask the Chatbot:")
if st.button("💡 Get Answer"):
    if user_query:
        answer = get_chatbot_response(user_query)
        st.markdown("### 🧠 Chatbot Answer:")
        st.success(answer)  
    else:
        st.warning("⚠ Please enter a question!")

# ✅ Footer
st.markdown("---")
st.write("💡 **Safalta Apki** - Your Guide to a Successful Career! 🚀")
st.write("👨‍💻 Developed by Team Safalta Apki")
