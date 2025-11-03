import streamlit as st
from dotenv import load_dotenv
import os
import traceback
from typing import List
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# LangChain 최신 1.0 API
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolSelectorMiddleware

# .env에서 OPENAI_API_KEY 불러오기
load_dotenv()
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
# api_key = os.getenv("OPENAI_API_KEY")

# Streamlit 페이지 설정
st.set_page_config(page_title="LangChain 1.0 Chatbot", page_icon="💬", layout="centered")
st.title("🤖 LangChain 1.0 + OpenAI Chatbot")

# -------------------------------
# 1️⃣ 도구 정의 (예시)
# -------------------------------
@tool
def get_current_time(timezone: str, location: str) -> str:
    """현재 시간을 지정된 타임존과 위치에 맞게 반환합니다."""
    import pytz
    from datetime import datetime
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f'{timezone} ({location}) 현재시각 {now}'
    except pytz.UnknownTimeZoneError:
        return f"알 수 없는 타임존: {timezone}"

@tool
def get_web_search(query: str, search_period: str) -> str:
    """DuckDuckGo API를 이용해 지정된 기간 내의 뉴스를 검색하여 결과를 반환합니다."""
    wrapper = DuckDuckGoSearchAPIWrapper(region="kr-kr", time=search_period)
    search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news", results_separator=';\n')
    return search.invoke(query)

tools = [get_current_time, get_web_search]

# -------------------------------
# 2️⃣ LLM 및 에이전트 생성
# -------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, api_key=api_key)

agent = create_agent(
    model=llm,
    tools=[get_current_time, get_web_search],
    middleware=[LLMToolSelectorMiddleware(max_tools=2)]
    )

# -------------------------------
# 3️⃣ Streamlit 세션 초기화m
# -------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage("저는 고성군청 직원을 위해 최선을 다하는 인공지능 도우미입니다. "),  
        AIMessage("무엇을 도와 드릴까요?")
    ]

# -------------------------------
# 4️⃣ 메시지 UI 표시
# -------------------------------
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# -------------------------------
# 5️⃣ 사용자 입력 및 응답 처리
# -------------------------------
user_input = st.chat_input("메시지를 입력하세요...")
if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            # LangChain 1.0 방식으로 invoke 실행
            response = agent.invoke({"input": user_input})
            ai_reply = response.get("output", "(응답 없음)")
            st.session_state.messages.append(AIMessage(content=ai_reply))
            message_placeholder.markdown(ai_reply)

        except Exception as e:
            st.error("❌ 오류 발생:")
            st.code(traceback.format_exc(), language="python")




#             for chunk in agent.stream({
#     "messages": [{"role": "user", "content": "Search for AI news and summarize the findings"}]
# }, stream_mode="values"):
#     # Each chunk contains the full state at that point
#     latest_message = chunk["messages"][-1]
#     if latest_message.content:
#         print(f"Agent: {latest_message.content}")
#     elif latest_message.tool_calls:
#         print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")