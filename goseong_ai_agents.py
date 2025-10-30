import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage

from langchain_core.tools import tool
from datetime import datetime
import pytz

from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from dotenv import load_dotenv
import os

from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import operator
import tempfile
import traceback
import time

from langchain_classic.chains import RetrievalQA
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
   

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# 모델 초기화
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.4, # 정확도  0.0 ~ 1.0
    timeout=30,  # 30초 타임아웃
    max_retries=2 ) 


# 도구 함수 정의
@tool
def get_current_time(timezone: str, location: str) -> str:
    """현재 시각을 반환하는 함수."""
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        result = f'{timezone} ({location}) 현재시각 {now}'
        return result
    except pytz.UnknownTimeZoneError:
        return f"알 수 없는 타임존: {timezone}"


@tool
def get_web_search(query: str, search_period: str) -> str:	
	
    """
    웹 검색을 수행하는 함수.
    """
    wrapper = DuckDuckGoSearchAPIWrapper(
        region="kr-kr", time=search_period)
    search = DuckDuckGoSearchResults(
        api_wrapper=wrapper,
        source="news",
        results_separator=';\n')
    
    docs = search.invoke(query)
    return docs


# 도구 바인딩
tools = [get_current_time, get_web_search]
tool_dict = {
    "get_current_time": get_current_time, 
    "get_web_search": get_web_search}
llm_with_tools = llm.bind_tools(tools)

# 사용자의 메시지 처리하기 위한 함수
def get_ai_response(messages: List[SystemMessage|HumanMessage|AIMessage|ToolMessage]):
    try:
        response_stream = llm_with_tools.stream(messages)
        gathered = ""
        for chunk in response_stream:
            yield chunk
            gathered += chunk.content

        # 도구 호출이 포함된 경우
        # (chunk.tool_calls 형태로 있는지 확인 필요)
        last = chunk  # 마지막 chunk
        if getattr(last, "tool_calls", None):
            st.session_state.messages.append(
                AIMessage(content=last.content, additional_kwargs={"tool_calls": last.tool_calls})
            )
            for tool_call in last.tool_calls:
                tool_id = tool_call.get("id") or tool_call.get("tool_call_id")
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})

                if not tool_id or not tool_name:
                    st.warning(f"⚠️ tool_call 데이터 이상: {tool_call}")
                    continue

                with st.spinner(f"🧩 도구 실행 중..."):
                    selected_tool = tool_dict[tool_name]
                    tool_result = selected_tool.invoke(tool_args)

                    tool_msg = ToolMessage(tool_call_id=tool_id, content=str(tool_result))
                    st.session_state.messages.append(tool_msg)
                    messages.append(tool_msg)

            # 재귀 호출
            for chunk in get_ai_response(messages):
                yield chunk
        else:
            if gathered and getattr(gathered, "content", None):
                st.session_state.messages.append(AIMessage(content=gathered.content))

    except Exception as e:
        st.error(f"❌ invoke() 호출 중 오류 발생: {e}")
        st.code(traceback.format_exc(), language="python")


def answer_question(query: str):
    st.write("🚀 질문 처리 시작")
    vectorstore = st.session_state.get("vectorstore")
    if vectorstore is None:
        st.warning("⚠️ PDF 학습이 아직 완료되지 않았습니다.")
        return "먼저 PDF 문서를 업로드하고 학습시켜 주세요."

    st.write("✅ vectorstore 확인 완료")
    try:
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            st.write(f"  문서 {i} 유사도: {score:.4f}")

        SIMILARITY_THRESHOLD = 1.1
        relevant_docs = [doc for doc, score in docs_with_scores if score < SIMILARITY_THRESHOLD]
        if not relevant_docs:
            st.warning("⚠️ 질문과 관련된 내용을 찾을 수 없습니다.")
            return "죄송합니다. 관련된 정보를 찾지 못했습니다."

        retriever = vectorstore.as_retriever(search_kwargs={"k":3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        result = qa_chain.invoke({"query": query})
        if isinstance(result, dict):
            return result.get("result", "답변을 생성할 수 없습니다.")
        else:
            return str(result)
    except Exception as e:
        st.error(f"❌ 오류 발생: {e}")
        st.code(traceback.format_exc(), language="python")
        return f"오류가 발생했습니다: {e}"
                


def process1_f(uploaded_files1):
    """PDF 파일을 학습하여 벡터스토어 생성"""
    
    # 파일 개수 체크
    if uploaded_files1 and len(uploaded_files1) > 3:
        st.error("❌ PDF는 최대 3개까지 업로드 가능합니다!")
        st.warning("⚠️ PDF파일을 3개만 선택하여 주세요!")
        return None  # 여기서 바로 return
    
    # 파일이 없는 경우
    if not uploaded_files1:
        st.warning("⚠️ PDF 파일을 업로드해주세요.")
        return None

    try:
        with st.spinner("📚 PDF 임베딩 및 벡터스토어 생성 중... 잠시만 기다려주세요"):
            all_splits = []
            
            # 각 PDF 파일 처리
            for idx, uploaded_file in enumerate(uploaded_files1, 1):
                st.write(f"📄 {idx}/{len(uploaded_files1)} 파일 처리 중: {uploaded_file.name}")
                
                # 임시 파일 생성
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                try:
                    # PDF 로드
                    loader = PyPDFLoader(tmp_path)
                    data = loader.load()
                    
                    # 청킹
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=300, 
                        chunk_overlap=50
                    )
                    splits = splitter.split_documents(data)
                    all_splits.extend(splits)
                    
                    st.success(f"✅ {uploaded_file.name}: {len(splits)}개 문서로 분할")
                    
                finally:
                    # 임시 파일 삭제
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            # 총 청크 수 표시
            st.info(f"📊 총 문서 분할 수: {len(all_splits)}")

            # Embedding 생성
            embedding = OpenAIEmbeddings(
                model="text-embedding-3-large", 
                api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
            )
            
            # 저장 디렉토리 설정
            persist_directory = "c:/faiss_store"
            os.makedirs(persist_directory, exist_ok=True)

            # 배치 단위 임베딩
            batch_size = 20
            vectorstore = None
            total_batches = (len(all_splits) + batch_size - 1) // batch_size
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(0, len(all_splits), batch_size):
                batch = all_splits[i:i+batch_size]
                batch_num = i//batch_size + 1
                
                status_text.text(f"🔄 배치 {batch_num}/{total_batches} 학습자료 저장 중...")
                progress_bar.progress(batch_num / total_batches)
                
                try:
                    if vectorstore is None:
                        # 첫 배치로 vectorstore 생성
                        vectorstore = FAISS.from_documents(batch, embedding)
                    else:
                        # 기존 vectorstore에 추가
                        vectorstore.add_documents(batch)
                    
                    # 로컬에 저장
                    vectorstore.save_local(persist_directory)
                    time.sleep(1.5)  # API 레이트 리밋 방지
                    
                except Exception as e:
                    st.error(f"❌ 배치 {batch_num} 학습자료 저장 실패: {e}")
                    continue

            progress_bar.progress(1.0)
            status_text.text("✅ 학습자료 저장 완료!")
            
            st.success("🎉 학습이 완료되었습니다!")
            st.toast("학습한 문서를 바탕으로 질문해 보세요!", icon="🎉")
            return vectorstore
            
    except Exception as e:
        st.error(f"❌ 학습 중 오류 발생: {e}")
        st.code(traceback.format_exc(), language="python")
        return None
    

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
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

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
    st.session_state.messages.append(HumanMessage(prompt)) # 사용자 메시지 저장


        # vectorstore 존재 여부 확인
    vectorstore = st.session_state.get("vectorstore")
    
    if vectorstore is not None:
        # 벡터스토어 기반 답변
        with st.spinner("📚 학습된 문서를 검색하는 중..."):
            answer = answer_question(prompt)
        
        # 관련 문서가 없는 경우 일반 모드로 전환
        if answer and "죄송합니다. " in answer and len(answer) < 20:
            st.info("💡 학습된 문서에서 관련 내용을 찾지 못했습니다. 일반 AI 모드로 전환합니다.")
            st.write([type(m) for m in "messages"])
            response = get_ai_response(st.session_state["messages"])
            result = st.chat_message("assistant").write(response)
            st.write(1)
            st.session_state["messages"].append(AIMessage(result))
        else:
            # 문서 기반 답변
            st.write(answer)
            st.write(3)
            st.chat_message("assistant").write(answer)
            st.session_state.append(AIMessage(content=str(answer)))
    else:
        # 일반 AI 모드
        st.info("🤖 일반 AI 모드로 답변합니다. 문서를 학습하면 더 정확한 답변을 받을 수 있습니다.")
        st.write([type(m) for m in "messages"])
        response = get_ai_response(st.session_state["messages"])
        result = st.chat_message("assistant").write(response)
        st.session_state["messages"].append(AIMessage(content=str(result)))


# 문서 학습 함수 불러오기
if process1:
    st.session_state["vectorstore"] = process1_f(uploaded_files1)


