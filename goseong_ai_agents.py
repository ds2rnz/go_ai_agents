import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from langchain_core.tools import tool
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
    temperature=0.4, # 정확도  0.0 ~ 1.0
    timeout=10,  # 10초 타임아웃
    max_retries=2 ) 


# 도구 함수 정의
@tool
def get_current_time(timezone: str, location: str) -> str:
    """현재 시각을 반환하는 함수."""
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        result = f'{timezone} ({location}) 현재시각 {now}'
        print(result)
        return result
    except pytz.UnknownTimeZoneError:
        return f"알 수 없는 타임존: {timezone}"
    
@tool
def get_web_search(query: str, search_period: str) -> str:	
	
    """
    웹 검색을 수행하는 함수.

    Args:
        query (str): 검색어
        search_period (str): 검색 기간 (e.g., "w" for past week, "m" for past month, "y" for past year)	#②

    Returns:
        str: 검색 결과
    """
    wrapper = DuckDuckGoSearchAPIWrapper(region="kr-kr", time=search_period)

    print('-------- WEB SEARCH --------')
    print(query)
    print(search_period)

    search = DuckDuckGoSearchResults(
        api_wrapper=wrapper,
        # source="news",
        results_separator=';\n'
    )

    docs = search.invoke(query)
    return docs


# 도구 바인딩
tools = [get_current_time, get_web_search]
tool_dict = {
    "get_current_time": get_current_time, 
    "get_web_search": get_web_search
}

llm_with_tools = llm.bind_tools(tools)


# 사용자의 메시지 처리하기 위한 함수
def get_ai_response(messages):
    response = llm_with_tools.invoke(messages, tools=tools) 
    
    for chunk in response.content:
        if chunk.get("type") == "text":
            st.chat_message("assistant").write_stream(chunk["text"])
    return response.content             





# --- Streamlit 앱 설정 ---
st.set_page_config(page_title="AI Chat", page_icon="💬", layout="wide")

st.title("💬 고성군청 AI Chatbot 도우미")

# --- 화면 디자인 ---
st.markdown("""
    <style>
    /* 기본 바디 폰트 및 배경 */
    body {
        background-color: #f0f2f6;
        font-family: 'Noto Sans KR', sans-serif;
        color: #333;
    }

    /* 사이드바 배경과 그림자 */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: none;
        box-shadow: 2px 0 8px rgba(0, 0, 0, 0.1);
        padding: 1rem 1.5rem;
    }

    /* 사이드바 각 섹션 박스 스타일 */
    .sidebar-section {
        background: #fafafa;
        border-radius: 12px;
        padding: 20px 25px;
        margin-bottom: 25px;
        box-shadow: 0 3px 8px rgba(0,0,0,0.05);
        transition: box-shadow 0.3s ease;
    }

    .sidebar-section:hover {
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    /* 섹션 제목 스타일 */
    .sidebar-section h2, .sidebar-section h3 {
        font-weight: 700;
        color: #1f2937;  /* 어두운 네이비 톤 */
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* 아이콘 크기 조절 */
    .sidebar-section h2 svg, .sidebar-section h3 svg {
        width: 24px;
        height: 24px;
        fill: #3b82f6; /* 파란색 톤 */
    }

    /* 버튼 스타일 개선 */
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 10px;
        padding: 10px 18px;
        font-weight: 600;
        font-size: 16px;
        border: none;
        transition: background-color 0.25s ease;
        width: 100%;
        cursor: pointer;
    }

    </style>
""", unsafe_allow_html=True)



# 스트림릿 session_state에 메시지 저장

messages = [
        SystemMessage(content="저는 고성군청 직원을 위해 최선을 다하는 인공지능 도우미입니다. "),  
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
if prompt := st.chat_input():
    st.chat_message("user").write(prompt) # 사용자 메시지 출력
    messages.append(HumanMessage(prompt)) # 사용자 메시지 저장

    response = get_ai_response(messages)
    
    result = response # AI 메시지 출력
    messages.append(AIMessage(result)) # AI 메시지 저장    
