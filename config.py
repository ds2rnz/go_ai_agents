import streamlit as st
# from langchain.tools import tool
# from langchain.chat_models import init_chat_model
# from langchain.agents import create_agent
from datetime import datetime
import pytz
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import os
from langchain_openai import OpenAIEmbeddings



def embedding():

    OpenAIEmbeddings(
        model="text-embedding-3-large", 
        api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        )


