import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from datetime import datetime
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
import pytz
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from dotenv import load_dotenv
import os
from langchain.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver

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

load_dotenv()
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

ddg_search_tool = DuckDuckGoSearchRun()

checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "1"}}

llm = init_chat_model(
    model = "openai:gpt-4o-mini",
    temperature=0.5, 
    max_tokens=1000, 
    timeout=10, 
    max_retries=2, 
    )

agent = create_agent(
    model=llm,
    tools=[get_current_time, ddg_search_tool],
    middleware=[],
    checkpointer=checkpointer,
)


# --- Streamlit 앱 설정 ---
st.set_page_config(page_title="GPT AI 도우미", page_icon="💬", layout="wide")
st.title("💬 고성군청 AI 도우미")

# --- 화면 디자인 ---
st.markdown("""
    <style>
    /* CSS 스타일은 그대로 */
    </style>
""", unsafe_allow_html=True)

# 스트림릿 session_state에 메시지 저장
messages = [
        {"role": "system", "content": "저는 고성군청 직원을 위해 최선을 다하는 인공지능 도우미입니다."},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "무엇이을 도와 드릴까요?"}
]

# 스트림릿 화면에 메시지 출력
for msg in messages:
    if msg:
        if isinstance(msg, SystemMessage):
            st.chat_message("system").write((msg.content))
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg['messages'][2].content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(HumanMessage(msg['messages'][-1].content))

# 사용자 입력 처리
if prompt := st.chat_input(placeholder="무엇이든 물어보세요?"):
    st.chat_message("user").write(prompt)  # 사용자 메시지 출력
    messages.append(HumanMessage(prompt))  # 사용자 메시지 저장
    response = agent.invoke({"messages":[{"role":"user", "content":prompt}]}
                               config=config,
                               tool_choice='any' # 도구 사용 강제(일반 llm으로의 fallback 방지)  # AI 응답 처리
    messages.append(AIMessage(response['messages'][-1].content))  # AI 메시지 저장
    st.chat_message("assistant").write(response['messages'][-1].content)  # AI 응답 출력
    st.write(messages)
