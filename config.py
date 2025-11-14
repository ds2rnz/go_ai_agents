import streamlit as st
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from datetime import datetime
import pytz
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import os
from langchain_openai import OpenAIEmbeddings


@tool
def get_current_time(timezone: str, location: str) -> str:
    '''  해당 지역 현재시각을 구하는 함수 '''
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        result = f'{timezone} ({location}) 현재시각 {now}'
        return result
    except pytz.UnknownTimeZoneError:
        return f"알 수 없는 타임존: {timezone}"  

@tool
def get_web_search(query: str) -> str:
    """
    웹 검색을 수행하는 함수.

    Args:
        query (str): 검색어
    Returns:
        str: 검색 결과
    """
    custom_wrapper = DuckDuckGoSearchAPIWrapper(region="kr-kr", time="y", max_results=10)
    search = DuckDuckGoSearchResults(
        api_wrapper=custom_wrapper,
        source="news, image, text",
        results_separator=';\n')
    
    results = search.run(query)

    st.toast("웹 검색을 통아여 알아보고 있습니다.", icon="🎉")
    return results



def embedding():

    OpenAIEmbeddings(
        model="text-embedding-3-large", 
        api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        )


def llm():
    init_chat_model(
        model = "openai:gpt-4o",
        temperature=0.6, 
        max_tokens=1000, 
        timeout=10, 
        max_retries=2, 
        )
    
def agent():
    create_agent(
        model=llm,
        tools=[get_current_time, get_web_search],
        middleware=[],
        system_prompt="사용자가 질문을하면 구체적이고 자세하게 설명해주고 모르는 내용이면 인터넷 검색을 꼭해서 답변해줘 그리고 한글로 답해주세요", 
        )
