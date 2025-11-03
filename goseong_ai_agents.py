import streamlit as st
from dotenv import load_dotenv
import os
from typing import List
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

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
import pytz
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_core.vectorstores.base import VectorStore
import traceback
import inspect
import time
import base64
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# .env에서 OPENAI_API_KEY 불러오기
load_dotenv()
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
# api_key = os.getenv("OPENAI_API_KEY")


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

# -------------------------------
# 2️⃣ LLM 및 에이전트 생성
# -------------------------------

#tools = [get_current_time, get_web_search]
#tool_dict = {tool.name: tool for tool in tools}
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, api_key=api_key)
agent = create_agent(
    model=llm,
    # tools=[get_current_time, get_web_search],
    # middleware=[LLMToolSelectorMiddleware(max_tools=2)]
    )


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


# def answer_question(query: str, timeout_sec: int = 60):
#     """LLM 기반 PDF QA """

#     st.write("🚀 질문 처리 시작")
#     start_time = time.time()

#     vectorstore = st.session_state.get("vectorstore")
#     if vectorstore is None:
#         st.warning("⚠️ PDF 학습이 아직 완료되지 않았습니다.")
#         return "먼저 PDF 문서를 업로드하고 학습시켜 주세요."

#     st.write("✅ vectorstore 확인 완료")

#     try:
#         # 문서에서 유사도 검사
#         docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)
        
#         st.write(f"🔍 문서 검색 횟수: {len(docs_with_scores)}회")
        
#         # 디버깅: 유사도 점수 표시
#         for i, (doc, score) in enumerate(docs_with_scores, 1):
#             st.write(f"  문서 {i} 유사도: {score:.4f}")
        
#         # 유사도 임계값 설정
#         SIMILARITY_THRESHOLD = 0.9
        
#         relevant_docs = [doc for doc, score in docs_with_scores if score < SIMILARITY_THRESHOLD]
        
#         if not relevant_docs:
#             st.warning(f"⚠️ 질문과 관련된 내용을 찾을 수 없습니다. (최소 유사도: {min(score for _, score in docs_with_scores):.4f})")
#             return "죄송합니다. "
        
#         st.success(f"✅ {len(relevant_docs)}개의 관련 문서를 찾았습니다!")

#         # Retriever 생성 
#         retriever = vectorstore.as_retriever(
#             search_type="similarity", 
#             search_kwargs={"k": 3}
#         )
#         st.write("✅ retriever 생성 완료")

       
#         # QA Chain 생성
#         qa_chain = create_retriever_tool(
#             retriever=retriever,
#             name="document_search",
#             description="문서 기반 질의응답을 수행합니다."
#             )
#         st.write("✅ 유사도 연결 생성 완료")

#         # 질문 실행
#         try:
#             with st.spinner("🤔 답변 생성 중..."):
#                 result = qa_chain.invoke({"query": query})
#         except Exception as e:
#             st.error(f"❌ invoke() 호출 중 오류 발생: {e}")
#             st.code(traceback.format_exc(), language="python")
#             return f"오류가 발생했습니다: {e}"
        
#         elapsed = time.time() - start_time
#         st.success(f"✅ 응답 완료 ({elapsed:.2f}초)")

#         # 결과 추출
#         if isinstance(result, dict):
#             answer = result.get("result", "답변을 생성할 수 없습니다.")
            
#             # LLM이 "관련 정보 없음"이라고 답한 경우 감지
#             if "관련 정보를 찾을 수 없습니다" in answer or "관련이 없" in answer:
#                 st.info("💡 학습된 문서와 질문이 관련이 없는 것 같습니다.")
            
#             # 출처 문서 표시 (선택사항)
#             if result.get("source_documents"):
#                 with st.expander("📚 참고 문서 보기"):
#                     for i, doc in enumerate(result["source_documents"], 1):
#                         st.text_area(f"문서 {i}", doc.page_content[:300], height=200)
            
#             return answer
#         else:
#             return str(result)

#     except Exception as e:
#         st.error(f"❌ 오류 발생: {e}")
#         st.code(traceback.format_exc(), language="python")
#         return f"오류가 발생했습니다: {e}"
    

# def process1_f(uploaded_files1):
#     """PDF 파일을 학습하여 벡터스토어 생성"""
    
#     # 파일 개수 체크
#     if uploaded_files1 and len(uploaded_files1) > 3:
#         st.error("❌ PDF는 최대 3개까지 업로드 가능합니다!")
#         st.warning("⚠️ PDF파일을 3개만 선택하여 주세요!")
#         return None  # 여기서 바로 return
    
#     # 파일이 없는 경우
#     if not uploaded_files1:
#         st.warning("⚠️ PDF 파일을 업로드해주세요.")
#         return None

#     try:
#         with st.spinner("📚 PDF 임베딩 및 벡터스토어 생성 중... 잠시만 기다려주세요"):
#             all_splits = []
            
#             # 각 PDF 파일 처리
#             for idx, uploaded_file in enumerate(uploaded_files1, 1):
#                 st.write(f"📄 {idx}/{len(uploaded_files1)} 파일 처리 중: {uploaded_file.name}")
                
#                 # 임시 파일 생성
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#                     tmp_file.write(uploaded_file.read())
#                     tmp_path = tmp_file.name

#                 try:
#                     # PDF 로드
#                     loader = PyPDFLoader(tmp_path)
#                     data = loader.load()
                    
#                     # 청킹
#                     splitter = RecursiveCharacterTextSplitter(
#                         chunk_size=300, 
#                         chunk_overlap=50
#                     )
#                     splits = splitter.split_documents(data)
#                     all_splits.extend(splits)
                    
#                     st.success(f"✅ {uploaded_file.name}: {len(splits)}개 문서로 분할")
                    
#                 finally:
#                     # 임시 파일 삭제
#                     if os.path.exists(tmp_path):
#                         os.remove(tmp_path)

#             # 총 청크 수 표시
#             st.info(f"📊 총 문서 분할 수: {len(all_splits)}")

#             # Embedding 생성
#             embedding = OpenAIEmbeddings(
#                 model="text-embedding-3-large", 
#                 api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
#             )
            
#             # 저장 디렉토리 설정
#             persist_directory = "c:/faiss_store"
#             os.makedirs(persist_directory, exist_ok=True)

#             # 배치 단위 임베딩
#             batch_size = 20
#             vectorstore = None
#             total_batches = (len(all_splits) + batch_size - 1) // batch_size
            
#             progress_bar = st.progress(0)
#             status_text = st.empty()
            
#             for i in range(0, len(all_splits), batch_size):
#                 batch = all_splits[i:i+batch_size]
#                 batch_num = i//batch_size + 1
                
#                 status_text.text(f"🔄 배치 {batch_num}/{total_batches} 학습자료 저장 중...")
#                 progress_bar.progress(batch_num / total_batches)
                
#                 try:
#                     if vectorstore is None:
#                         # 첫 배치로 vectorstore 생성
#                         vectorstore = FAISS.from_documents(batch, embedding)
#                     else:
#                         # 기존 vectorstore에 추가
#                         vectorstore.add_documents(batch)
                    
#                     # 로컬에 저장
#                     vectorstore.save_local(persist_directory)
#                     time.sleep(1.5)  # API 레이트 리밋 방지
                    
#                 except Exception as e:
#                     st.error(f"❌ 배치 {batch_num} 학습자료 저장 실패: {e}")
#                     continue

#             progress_bar.progress(1.0)
#             status_text.text("✅ 학습자료 저장 완료!")
            
#             st.success("🎉 학습이 완료되었습니다!")
#             # st.balloons()
#             st.toast("학습한 문서를 바탕으로 질문해 보세요!", icon="🎉")
#             return vectorstore
            
#     except Exception as e:
#         st.error(f"❌ 학습 중 오류 발생: {e}")
#         st.code(traceback.format_exc(), language="python")
#         return None    



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




with st.sidebar:
    st.header("⚙️ 문서 :red[학습기]")
    uploaded_files1 = st.file_uploader(
    "📎 학습 문서 업로드 창 :red[PDF]파일  :red[3]개만 가능", type=['pdf'], accept_multiple_files=True
    )
    process1 = st.button("🚀 학습시작",        
            type = "primary",
            disabled=(uploaded_files1 is None))

    st.markdown("---")
    st.markdown("### 📖 :blue[사용방법]")
    st.markdown("""
        1. PDF 파일을 업로드하세요(최대 3개만)
        2. "학습시작"  버튼을 클릭하세요
        3. 학습한 문서를 바탕으로 사용자 요청에 따라
        답변합니다. 
        """)
        
    st.markdown("---")

    
       

# 스트림릿 session_state에 메시지 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage(content="저는 고성군청 직원을 위해 최선을 다하는 인공지능 도우미입니다. "),  
        AIMessage(content="무엇을 도와 드릴까요?")
    ]

# 학습 data가 없으면 초기화
# if "vectorstore" not in st.session_state:
#     st.session_state["vectorstore"] = None

# 스트림릿 화면에 메시지 출력
for msg in st.session_state["messages"]:
    if msg.content:
        if isinstance(msg, SystemMessage):
            st.chat_message("system").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        # elif isinstance(msg, ToolMessage):
        #     st.chat_message("tool").write(msg.content)


# 사용자 입력 처리
if prompt := st.chat_input(placeholder = "무엇이든 물어보세요?"):
    st.chat_message("user").write(prompt) # 사용자 메시지 출력
    st.session_state["messages"].append(HumanMessage(prompt)) # 사용자 메시지 저장

response = agent.stream(prompt)
st.session_state["message"].append(AIMessage(response))
st.chat_message("assistant").write_stream(response)

# user_input = st.chat_input("메시지를 입력하세요...")
# if user_input:
#     st.session_state.messages.append(HumanMessage(content=user_input))
#     with st.chat_message("user"):
#         st.markdown(user_input)

#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         try:
#             # LangChain 1.0 방식으로 invoke 실행
#             response = agent.invoke({
#                     "messages": [HumanMessage(content=user_input)]
#                      })
#             ai_reply = response.get("output", "(응답 없음)")
#             st.session_state.messages.append(AIMessage(content=ai_reply))
#             message_placeholder.markdown(ai_reply)

#         except Exception as e:
#             st.error("❌ 오류 발생:")
#             st.code(traceback.format_exc(), language="python")


        # vectorstore 존재 여부 확인
#    vectorstore = st.session_state.get("vectorstore")
    
#    if vectorstore is not None:
#       # 벡터스토어 기반 답변
#        with st.spinner("📚 학습된 문서를 검색하는 중..."):
#            answer = answer_question(prompt)
        
        # 관련 문서가 없는 경우 일반 모드로 전환
    #     if answer and "죄송합니다. " in answer and len(answer) < 20:
    #         st.info("💡 학습된 문서에서 관련 내용을 찾지 못했습니다. 일반 AI 모드로 전환합니다.")
    #         # st.write([type(m) for m in "messages"])
    #         response = get_ai_response(st.session_state["messages"])
    #         result = st.chat_message("assistant").markdown(response)
    #         st.write(1)
    #         st.session_state["messages"].append(AIMessage(content=str(result)))
    #     else:
    #         # 문서 기반 답변
    #         st.write(answer)
    #         st.write(3)
    #         st.chat_message("assistant").write(answer)
    #         st.session_state.append(AIMessage(content=str(answer)))
    # else:
    #     # 일반 AI 모드
#         st.info("🤖 일반 AI 모드로 답변합니다. 문서를 학습하면 더 정확한 답변을 받을 수 있습니다.")
#         st.write([type(m) for m in "messages"])
#         response = get_ai_response(st.session_state["messages"])
#         result = st.chat_message("assistant").write(response)
#         st.session_state["messages"].append(AIMessage(content=str(result)))


# # 문서 학습 함수 불러오기
# if process1:
#     st.session_state["vectorstore"] = process1_f(uploaded_files1)




# '''
# # -------------------------------
# # 3️⃣ Streamlit 세션 초기화m
# # -------------------------------
# # if "messages" not in st.session_state:
# #     st.session_state["messages"] = [
# #         SystemMessage("저는 고성군청 직원을 위해 최선을 다하는 인공지능 도우미입니다. "),  
# #         AIMessage("무엇을 도와 드릴까요?")
# #     ]

# # -------------------------------
# # 4️⃣ 메시지 UI 표시
# # -------------------------------
# # for msg in st.session_state.messages:
# #     role = "user" if isinstance(msg, HumanMessage) else "assistant"
# #     with st.chat_message(role):
# #         st.markdown(msg.content)

# # -------------------------------
# # 5️⃣ 사용자 입력 및 응답 처리
# # -------------------------------
# user_input = st.chat_input("메시지를 입력하세요...")
# if user_input:
#     st.session_state.messages.append(HumanMessage(content=user_input))
#     with st.chat_message("user"):
#         st.markdown(user_input)

#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         try:
#             # LangChain 1.0 방식으로 invoke 실행
#             response = agent.invoke({
#                     "messages": [HumanMessage(content=user_input)]
#                      })
#             ai_reply = response.get("output", "(응답 없음)")
#             st.session_state.messages.append(AIMessage(content=ai_reply))
#             message_placeholder.markdown(ai_reply)

#         except Exception as e:
#             st.error("❌ 오류 발생:")
#             st.code(traceback.format_exc(), language="python")




# #             for chunk in agent.stream({
# #     "messages": [{"role": "user", "content": "Search for AI news and summarize the findings"}]
# # }, stream_mode="values"):
# #     # Each chunk contains the full state at that point
# #     latest_message = chunk["messages"][-1]
# #     if latest_message.content:
# #         print(f"Agent: {latest_message.content}")
# #     elif latest_message.tool_calls:
# #         print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")
# '''