import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolSelectorMiddleware
from langchain_core.tools import tool
from langchain.messages import HumanMessage, AIMessage, SystemMessage
import datetime
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from dotenv import load_dotenv
import os



# -------------------------------
# 1️⃣ 환경 설정
# -------------------------------
load_dotenv()
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

llm = ChatOpenAI(
    model="gpt-4o-mini",  # 또는 gpt-4o
    temperature=0.4,
    api_key=api_key,
)

# -------------------------------
# 2️⃣ Agent 생성
# -------------------------------
agent = create_agent(model=llm)

# -------------------------------
# 3️⃣ Streamlit UI
# -------------------------------
st.set_page_config(page_title="LangChain Chatbot", page_icon="🤖")
st.title("🤖 LangChain create_agent() Chatbot")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage(content="저는 고성군청 직원을 위해 최선을 다하는 인공지능 도우미입니다."),
        AIMessage(content="안녕하세요! 무엇을 도와드릴까요? 😊"),
    ]

# 기존 대화 표시
for msg in st.session_state["messages"]:
    role = (
        "assistant" if isinstance(msg, AIMessage)
        else "user" if isinstance(msg, HumanMessage)
        else "system"
    )
    st.chat_message(role).write(msg.content)

# -------------------------------
# 4️⃣ 사용자 입력 처리
# -------------------------------
if prompt := st.chat_input("무엇이든 물어보세요!"):
    # 사용자 입력 저장 및 표시
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append(HumanMessage(content=prompt))

    # 스트리밍 응답용 placeholder
    with st.chat_message("assistant"):
        stream_area = st.empty()
        streamed_text = ""

        # 메시지 기반으로 agent 호출
        for event in llm.stream({"messages": st.session_state["messages"]}):
            # event는 {"messages": [...]} 형태로 옴
            if "messages" in event:
                msg = event["messages"][-1]
                st.write(msg)
                if isinstance(msg, AIMessage):
                    streamed_text += msg.content
                    st.markdown(streamed_text + "▌")

        # 마지막 응답 표시
        stream_area.markdown(streamed_text)
        st.session_state["messages"].append(AIMessage(content=streamed_text))

