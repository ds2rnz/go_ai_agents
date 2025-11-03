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


# .env에서 OPENAI_API_KEY 불러오기
load_dotenv()
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
# api_key = os.getenv("OPENAI_API_KEY")

# -------------------------------
# 1️⃣ 간단한 도구 정의
# -------------------------------

# # -------------------------------
# # 1️⃣ 도구 정의
# # -------------------------------
# @tool
# def get_current_time(timezone: str, location: str) -> str:
#     """현재 시간을 지정된 타임존과 위치에 맞게 반환합니다."""
#     import pytz
#     from datetime import datetime
#     try:
#         tz = pytz.timezone(timezone)
#         now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
#         return f'{timezone} ({location}) 현재시각 {now}'
#     except pytz.UnknownTimeZoneError:
#         return f"알 수 없는 타임존: {timezone}"

# @tool
# def get_web_search(query: str, search_period: str) -> str:
#     """DuckDuckGo API를 이용해 지정된 기간 내의 뉴스를 검색하여 결과를 반환합니다."""
#     wrapper = DuckDuckGoSearchAPIWrapper(region="kr-kr", time=search_period)
#     search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news", results_separator=';\n')
#     return search.invoke(query)

# def get_current_time(query: str = "") -> str:
#     """현재 시간을 반환하는 도구"""
#     now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     return f"현재 시간은 {now} 입니다."

# def get_web_search(query: str) -> str:
#     """가짜 웹 검색 예시 (실제 검색은 아님)"""
#     return f"'{query}'에 대한 웹 검색 결과를 찾을 수 없습니다."

# LangChain의 Tool 객체로 등록
#tools = [get_current_time, get_web_search]

# -------------------------------
# 2️⃣ LLM 초기화
# -------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",   # 또는 gpt-4o, gpt-3.5-turbo 등
    temperature=0.4,
)

# -------------------------------
# 3️⃣ Agent 생성
# -------------------------------
agent = create_agent(
    model=llm,
#    tools=tools,
#    middleware=[LLMToolSelectorMiddleware(max_tools=2)],
)

# -------------------------------
# 4️⃣ Streamlit UI 설정
# -------------------------------
st.set_page_config(page_title="LangChain Chatbot", page_icon="🤖")
st.title("🤖 LangChain create_agent() Chatbot")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage(content="저는 고성군청 직원을 위해 최선을 다하는 인공지능 도우미입니다."),
        AIMessage(content="안녕하세요! 무엇을 도와드릴까요? 😊"),
    ]

# 기존 대화 기록 출력
for msg in st.session_state["messages"]:
    if isinstance(msg, SystemMessage):
        st.chat_message("system").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# -------------------------------
# 5️⃣ 사용자 입력 처리
# -------------------------------
if prompt := st.chat_input("무엇이든 물어보세요!"):
    # 사용자 메시지 출력
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append(HumanMessage(content=prompt))

    # Agent 호출
    # response = agent.invoke({
    #     "messages": [HumanMessage(content=prompt)]
    # })
    for chunk in agent.stream({"messages": prompt}, stream_mode="values"):
        if "messages" in chunk and chunk["messages"]:
            response = chunk["messages"][-1]
            st.write(response)
 #           st.write(type(response))
#
    # 응답 내용 추출
    if isinstance(response, dict) and "messages" in response:
        msg = response["messages"][-1]
        st.write(msg)
        content = msg.content if hasattr(msg, "content") else str(msg)
        st.write(content)
    else:
        content = str(response)

    st.chat_message("assistant").write(msg[AIMessage.content][-1])

    # AI 응답 출력
    # st.chat_message("assistant").write(f"message:{ai_reply['messages'][-1].content}")
    # st.session_state["messages"].append(AIMessage(content=ai_reply))
    #st.chat_message("assistant").write(response['messages'][-1].content)
    st.session_state["messages"].append(AIMessage(content))
    #(f"Response: {result1['messages'][-1].content}")






# def get_ai_response(messages):
#     try:
#         response = agent.stream({"messages":messages})
#         response = {"message": messages}
#         gathered = None
#         for chunk in agent.stream(response, stream_mode="updates"):
#             yield chunk
#             if gathered is None:
#                 gathered = chunk
#             else:
#                 gathered += chunk

#         if gathered and getattr(gathered, "tool_calls", None):
#             st.session_state.messages.append(gathered)
#             for tool_call in gathered.tool_calls:
#                 selected_tool = tool_dict.get(tool_call['name'])
#                 if selected_tool:
#                     with st.spinner("도구 실행 중..."):
#                         try:
#                             tool_msg = selected_tool.invoke(tool_call)
#                             st.session_state.messages.append(tool_msg)
#                         except Exception as e:
#                             st.error(f"도구 실행 오류:{e}")
#             # 도구 호출 후 재귀적으로 응답 생성
#             yield from get_ai_response(st.session_state["messages"])


#          # AI의 최종 응답이 있으면 이를 출력
#         if gathered:
#             # gathered가 최종적으로 AI 응답을 포함하고 있으면 이를 출력
#             ai_response = gathered.get('content', '')  # AI 응답의 내용을 가져오기

#             if ai_response:
#                 # Streamlit에 AI의 응답 출력
#                 st.write(f"AI 응답: {ai_response}")
#                 # 필요시 UI를 통해 '대화' 형식으로 응답을 추가
#                 st.session_state.messages.append({"role": "ai", "content": ai_response})


#     except Exception as e:
#         st.error(f"❌ invoke() 호출 중 오류 발생: {e}")



