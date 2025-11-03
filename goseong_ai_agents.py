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



load_dotenv()
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# -------------------------------
# 2️⃣ LLM 설정
# -------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",  # 또는 "gpt-4o"
    temperature=0.4,
    api_key=api_key,
)

# -------------------------------
# 3️⃣ DuckDuckGo 검색 Tool 정의
# -------------------------------
@tool
def web_search(query: str) -> str:
    """
    DuckDuckGo 검색을 수행하는 도구입니다.
    Args:
        query (str): 검색어
    Returns:
        str: 검색 결과 텍스트 요약
    """
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
    search = DuckDuckGoSearchResults(api_wrapper=wrapper)
    results = search.run(query)
    return results

# 사용할 도구 목록
tools = [web_search]

# -------------------------------
# 4️⃣ Agent 생성
# -------------------------------
agent = create_agent(
    model=llm,
    tools=tools
)

# -------------------------------
# 5️⃣ Streamlit UI
# -------------------------------
st.set_page_config(page_title="LangChain Web Search Chatbot", page_icon="🌐")
st.title("🌐 LangChain + DuckDuckGo Chatbot")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage(content="저는 고성군청 직원을 위해 일하는 인공지능 도우미입니다. 필요한 정보를 신속하게 찾아드리겠습니다."),
        AIMessage(content="안녕하세요! 무엇을 도와드릴까요? 😊"),
    ]

# 대화 기록 표시
for msg in st.session_state["messages"]:
    role = (
        "assistant" if isinstance(msg, AIMessage)
        else "user" if isinstance(msg, HumanMessage)
        else "system"
    )
    st.chat_message(role).write(msg.content)

# -------------------------------
# 6️⃣ 사용자 입력 처리
# -------------------------------
if prompt := st.chat_input("무엇이든 물어보세요!"):
    # 사용자 메시지 저장 및 표시
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append(HumanMessage(content=prompt))

    # 스트리밍 응답 처리
    with st.chat_message("assistant"):
        stream_area = st.empty()
        streamed_text = ""

        # agent.stream()을 사용하여 실시간 출력
        for event in agent.stream({"messages": st.session_state["messages"]}):
            if "messages" in event:
                msg = event["messages"][-1]
                if isinstance(msg, AIMessage):
                    streamed_text += msg.content
                    stream_area.markdown(streamed_text + "▌")

        # 최종 답변 출력
        stream_area.markdown(streamed_text)
        st.session_state["messages"].append(AIMessage(content=streamed_text))

