import streamlit as st
import os
from dotenv import load_dotenv
from pprint import pprint
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

# 커스텀 tool 생성
@tool 
def calculator(num_1:int, num_2:int) -> int: # typehint는 Agent가 tool의 입출력 형식을 이해하는 데 도움을 줍니다. 안정적인 작동을 위해 반드시 작성하는게 좋습니다.
    """입력받은 두 수의 덧셈을 반환합니다.""" # docstring은 tool의 설명으로 사용됩니다. Agent가 tool을 선택하는 데 도움을 줍니다.
    return num_1 + num_2

load_dotenv()
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

ddg_search_tool = DuckDuckGoSearchRun()

# 모델 초기화
llm = init_chat_model(
    model = "openai:gpt-4o-mini",
    temperature=0.5, 
    max_tokens=2000, 
    timeout=10, 
    max_retries=2, 
    )

agent = create_agent(
    model=llm,
    tools=[calculator, ddg_search_tool],
    system_prompt="너는 친절한 도우미야",
    middleware=[],
)

model = ChatOpenAI(
    model="gpt-4o-mini",
)

tools = model.bind_tools([calculator])
opneai_tool = [{"type": "web_search"},]

messages = [
        SystemMessage("너는 사용자를 돕기 위해 최선을 다하는 인공지능 봇이다. "),  
        AIMessage("무엇이을 도와 드릴까요?"),
        HumanMessage("")
    ]


response = model.invoke("올해 1월 한국에서 개봉하는 영화는? 그리고 1 더하기 5는 얼마야?"
        , tools=opneai_tool )
# pprint(response.content)



for setp in response.content:
    if setp.get("type") == "text":
        st.chat_message("assistant").write(setp["text"])


messages.append(AIMessage(setp))

st.write(messages)
