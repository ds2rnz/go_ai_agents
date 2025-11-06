import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain.tools import tool
from datetime import datetime
import pytz
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# 모델 초기화
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.4, 
    timeout=10,  
    max_retries=2 
)

# 도구 함수 정의
@tool
def get_current_time(timezone: str, location: str) -> str:
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        result = f'{timezone} ({location}) 현재시각 {now}'
        return result
    except pytz.UnknownTimeZoneError:
        return f"알 수 없는 타임존: {timezone}"
    
@tool
def get_web_search(query: str, search_period: str) -> str:
    wrapper = DuckDuckGoSearchAPIWrapper(region="kr-kr", time=search_period)
    search = DuckDuckGoSearchResults(api_wrapper=wrapper, results_separator=';\n')
    docs = search.invoke(query)
    return docs

tools = [get_current_time, get_web_search]
tool_dict = [{"type": "web_search"},]

llm_with_tools = llm.bind_tools()

# 사용자의 메시지 처리하기 위한 함수
def get_ai_response(messages):
    response = llm_with_tools.invoke(messages, tools=tool_dict) 
    # 스트리밍 응답 처리
    if isinstance(response, dict) and "text" in response:
        st.chat_message("assistant").write(response["text"])
    return response

# --- Streamlit 앱 설정 ---
st.set_page_config(page_title="AI Chat", page_icon="💬", layout="wide")
st.title("💬 고성군청 AI Chatbot 도우미")

# --- 화면 디자인 ---
st.markdown("""
    <style>
    /* CSS 스타일은 그대로 */
    </style>
""", unsafe_allow_html=True)

# 스트림릿 session_state에 메시지 저장
messages = [
        SystemMessage(content="저는 고성군청 직원을 위해 최선을 다하는 인공지능 도우미입니다."),
        AIMessage(content="무엇을 도와 드릴까요?")
]

# 스트림릿 화면에 메시지 출력
for msg in messages:
    if msg:
        if isinstance(msg, SystemMessage):
            st.chat_message("system").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)

# 사용자 입력 처리
if prompt := st.chat_input(placeholder="무엇이든 물어보세요?"):
    st.chat_message("user").write(prompt)  # 사용자 메시지 출력
    messages.append(HumanMessage(prompt))  # 사용자 메시지 저장
    response = get_ai_response(messages)  # AI 응답 처리
    result = response.get("text", "응답을 받지 못했습니다.")  # 응답 텍스트 추출
    messages.append(AIMessage(result))  # AI 메시지 저장
    st.chat_message("assistant").write(result)  # AI 응답 출력
