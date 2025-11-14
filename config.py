import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings


def embedding():
    OpenAIEmbeddings(
        model="text-embedding-3-large", 
        api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        )


