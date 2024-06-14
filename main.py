import streamlit as st
import openai
from config import OPENAI_API_KEY
from langchain.llms import OpenAI as LangchainOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Set the OpenAI API key for the openai library
openai.api_key = OPENAI_API_KEY

# Initialize the LangChain OpenAI client
langchain_openai_client = LangchainOpenAI(api_key=OPENAI_API_KEY, temperature=0.5)

# Define a simple translation prompt template
translation_template = PromptTemplate(
    template="Translate the following text to {target_language}: {text}",
    input_variables=["text", "target_language"]
)

# Create a LangChain with the defined template and language model
translation_chain = LLMChain(prompt=translation_template, llm=langchain_openai_client)

# Streamlit UI
st.title("Prompt Application")

# Sidebar for language selection
language = st.sidebar.selectbox("Select Language", ["English", "French", "Spanish"])

# Main content area
st.subheader("Enter your prompt:")
prompt = st.text_area("")

def translate_prompt(text, target_language):
    # Generate translated text using LangChain
    translated_prompt = translation_chain.run({"text": text, "target_language": target_language})
    return translated_prompt

if st.button("Generate Text"):
    if language == "English":
        target_language = "en"
    elif language == "French":
        target_language = "fr"
    elif language == "Spanish":
        target_language = "es"

    # Translate prompt if necessary
    if target_language != "en":
        translated_prompt = translate_prompt(prompt, target_language)
    else:
        translated_prompt = prompt

    # Generate text based on translated prompt using OpenAI's new ChatCompletion method
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": translated_prompt}
        ],
        max_tokens=150,
        temperature=0.7
    )

    generated_text = response['choices'][0]['message']['content'].strip()

    st.subheader("Generated Text:")
    st.write(generated_text)