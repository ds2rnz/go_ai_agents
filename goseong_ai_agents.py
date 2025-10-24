import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from datetime import datetime

import pytz
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from dotenv import load_dotenv
import os

from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import ast
import operator

from langchain_community.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from openai import OpenAI

import concurrent.futures
import traceback
import inspect
import time
import base64
from langchain import AIMessage

# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


OPENAI_API_KEY = "OPENAI_API_KEY"


client = OpenAI()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.4, # 정확도  0.0 ~ 1.0
    timeout=30,  # 30초 타임아웃
    max_retries=2 ) 

# langgraph 클래스 정의
class GraphState(TypedDict):
    messages: Annotated[list, operator.add]
    pdf_path: str
    pdf_content: str
    chunks: List[str]
    analysis_result: str


# 개발시 필요한 함수 / 에러 추출
# def debug_wrap(func):
#     """함수 실행 시 에러나 중단점을 추적하기 위한 디버깅 래퍼"""
#     def wrapper(*args, **kwargs):
#         func_name = func.__name__
#         try:
#             print(f"[DEBUG] ▶ 실행 시작: {func_name}")
#             result = func(*args, **kwargs)
#             print(f"[DEBUG] ✅ 실행 성공: {func_name}")
#             return result
#         except Exception as e:
#             tb = traceback.format_exc()
#             print(f"\n[ERROR] ❌ 함수 '{func_name}' 에서 예외 발생:")
#             print(f"  └─ {e}")
#             print(tb)
#             st.error(f"❌ 함수 '{func_name}' 실행 중 오류 발생: {e}")
#             st.code(tb, language="python")
#             raise
#     return wrapper


# PDF 처리 함수들 
def load_pdf_node(state: GraphState) -> GraphState:
    pdf_path = state.get("pdf_path", "")
    if not pdf_path or not os.path.exists(pdf_path):
        return {
            "pdf_content": "",
            "messages": [AIMessage(content=f"파일이 없거나 경로가 잘못되었습니다: {pdf_path}")]
        }
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        full_text = "\n\n".join([page.page_content for page in pages])
        return {
            "pdf_content": full_text,
            "messages": [AIMessage(content=f"✅ PDF 로드 완료: {len(pages)}페이지, {len(full_text):,}자")]
        }
    except Exception as e:
        return {
            "pdf_content": "",
            "messages": [AIMessage(content=f"❌ PDF 로드 실패: {str(e)}")]
         }


def chunk_pdf_node(state: GraphState) -> GraphState:
    """PDF 내용을 청크로 분할하는 노드"""
    content = state.get("pdf_content", "")
    
    if not content:
        return {"chunks": [],
            "messages": [AIMessage(content="분할할 내용이 없습니다.")]
        }
    
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        chunks = text_splitter.split_text(content)
        
        return {"chunks": chunks,
            "messages": [AIMessage(content=f"✅ 텍스트 분할 완료: {len(chunks)}개 청크")]
        }
    except Exception as e:
        return {"chunks": [],
            "messages": [AIMessage(content=f"❌ 텍스트 분할 실패: {str(e)}")]
        }



def analyze_pdf_node(state: GraphState) -> GraphState:
    content = state.get("pdf_content", "")
    chunks = state.get("chunks", [])
    if not content:
        return {"analysis_result": "", "messages": [AIMessage(content="분석할 내용이 없습니다.")]}
    try:
        words = [w for w in content.lower().split() if len(w) > 3]
        word_freq = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        analysis = {
            "word_count": len(words),
            "char_count": len(content),
            "chunk_count": len(chunks),
            "keywords": top_keywords,
            "preview": content[:500]
        }
        return {
            "analysis_result": str(analysis),
            "messages": [AIMessage(content="✅ PDF 분석 완료")]
        }
    except Exception as e:
        return {"analysis_result": "", "messages": [AIMessage(content=f"❌ 분석 실패: {str(e)}")]}
 
 
@st.cache_resource
def create_pdf_analysis_graph():
    """PDF 분석 그래프를 생성하는 함수"""
    graph = StateGraph(GraphState)
    graph.add_node("load_pdf", load_pdf_node)
    graph.add_node("chunk_pdf", chunk_pdf_node)
    graph.add_node("analyze_pdf", analyze_pdf_node)
    graph.set_entry_point("load_pdf")
    graph.add_edge("load_pdf", "chunk_pdf")
    graph.add_edge("chunk_pdf", "analyze_pdf")
    graph.add_edge("analyze_pdf", END)
    return graph.compile()


# 도구 정의
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
tool_dict = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools) # tool 사용 llm 정의


# @debug_wrap / 에러 확인 함수 요청
def get_ai_response(messages):
    response = llm_with_tools.stream(messages)
    gathered = None
    for chunk in response:
        yield chunk
        if gathered is None:
            gathered = chunk
        else:
            gathered += chunk

    if gathered and getattr(gathered, "tool_calls", None):
        st.session_state.messages.append(gathered)
        for tool_call in gathered.tool_calls:
            selected_tool = tool_dict.get(tool_call['name'])
            if selected_tool:
                with st.spinner("도구 실행 중..."):
                    tool_msg = selected_tool.invoke(tool_call)
                    st.session_state.messages.append(tool_msg)
        # 도구 호출 후 재귀적으로 응답 생성
        yield from get_ai_response(st.session_state.messages)


# @debug_wrap / 에러 확인 함수 요청
def answer_question(query: str, timeout_sec: int = 60):
    """LLM 기반 PDF QA - ThreadExecutor 제거한 안정적인 버전"""

    st.write("🚀 질문 처리 시작")
    start_time = time.time()

    vectorstore = st.session_state.get("vectorstore")
    if vectorstore is None:
        st.warning("⚠️ PDF 학습이 아직 완료되지 않았습니다.")
        return "먼저 PDF 문서를 업로드하고 학습시켜 주세요."

    st.write("✅ vectorstore 확인 완료")

    try:
        # 문서에서 유사도 검사
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)
        
        st.write(f"🔍 문서 검색 횟수: {len(docs_with_scores)}회")
        
        # 디버깅: 유사도 점수 표시
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            st.write(f"  문서 {i} 유사도: {score:.4f}")
        
        # 유사도 임계값 설정
        SIMILARITY_THRESHOLD = 1.1
        
        relevant_docs = [doc for doc, score in docs_with_scores if score < SIMILARITY_THRESHOLD]
        
        if not relevant_docs:
            st.warning(f"⚠️ 질문과 관련된 내용을 찾을 수 없습니다. (최소 유사도: {min(score for _, score in docs_with_scores):.4f})")
            return "죄송합니다. "
        
        st.success(f"✅ {len(relevant_docs)}개의 관련 문서를 찾았습니다!")

        # Retriever 생성 
        retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3}
        )
        st.write("✅ retriever 생성 완료")

       
        # QA Chain 생성
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,  # llm 가져오기
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            )
        st.write("✅ 유사도 연결 생성 완료")

        # 질문 실행
        try:
            with st.spinner("🤔 답변 생성 중..."):
                result = qa_chain.invoke({"query": query})
        except Exception as e:
            st.error(f"❌ invoke() 호출 중 오류 발생: {e}")
            st.code(traceback.format_exc(), language="python")
            return f"오류가 발생했습니다: {e}"
        
        elapsed = time.time() - start_time
        st.success(f"✅ 응답 완료 ({elapsed:.2f}초)")

        # 결과 추출
        if isinstance(result, dict):
            answer = result.get("result", "답변을 생성할 수 없습니다.")
            
            # LLM이 "관련 정보 없음"이라고 답한 경우 감지
            if "관련 정보를 찾을 수 없습니다" in answer or "관련이 없" in answer:
                st.info("💡 학습된 문서와 질문이 관련이 없는 것 같습니다.")
            
            # 출처 문서 표시 (선택사항)
            if result.get("source_documents"):
                with st.expander("📚 참고 문서 보기"):
                    for i, doc in enumerate(result["source_documents"], 1):
                        st.text_area(f"문서 {i}", doc.page_content[:300], height=200)
            
            return answer
        else:
            return str(result)

    except Exception as e:
        st.error(f"❌ 오류 발생: {e}")
        st.code(traceback.format_exc(), language="python")
        return f"오류가 발생했습니다: {e}"
    


# @debug_wrap / 에러 확인 함수 요청
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
                api_key=OPENAI_API_KEY
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
            # st.balloons()
            st.toast("학습한 문서를 바탕으로 질문해 보세요!", icon="🎉")
            return vectorstore
            
    except Exception as e:
        st.error(f"❌ 학습 중 오류 발생: {e}")
        st.code(traceback.format_exc(), language="python")
        return None



# @debug_wrap / 에러 확인 함수 요청
def process2_f(uploaded_files2):
    if not uploaded_files2:
        st.info("👆 PDF 파일을 업로드하여 시작하세요!")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_files2.read())
        tmp_path = tmp_file.name

    try:
        with st.spinner("📄 PDF 분석 중..."):
            app = create_pdf_analysis_graph()
            initial_state = {
                "messages": [],
                "pdf_path": tmp_path,
                "pdf_content": "",
                "chunks": [],
                "analysis_result": ""
            }
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("📄 PDF 로딩 중...")
            progress_bar.progress(33)
            result = app.invoke(initial_state)
            progress_bar.progress(100)
            status_text.text("✅ 분석 완료!")

            st.success("✅ PDF 분석이 완료되었습니다!")

            # 분석 결과 표시
            analysis_data = ast.literal_eval(result.get("analysis_result", "{}"))

            tab1, tab2, tab3 = st.tabs(["📊 분석 결과", "📝 키워드", "🔍 미리보기"])
            with tab1:
                col1, col2, col3 = st.columns(3)
                col1.metric("총 단어 수", f"{analysis_data.get('word_count', 0):,}")
                col2.metric("총 문자 수", f"{analysis_data.get('char_count', 0):,}")
                col3.metric("청크 수", analysis_data.get('chunk_count', 0))

            with tab2:
                keywords = analysis_data.get('keywords', [])
                if keywords:
                    import pandas as pd
                    df = pd.DataFrame(keywords, columns=["키워드", "빈도"])
                    st.dataframe(df, use_container_width=True)
                    st.bar_chart(df.set_index("키워드"))
                else:
                    st.info("키워드를 찾을 수 없습니다.")

            with tab3:
                st.text_area("문서 미리보기 (첫 500자)", analysis_data.get('preview', ''), height=300)

            with st.expander("🔧 처리 로그"):
                for i, msg in enumerate(result.get("messages", []), 1):
                    st.text(f"{i}. {msg.content}")

            with st.expander("🐛 디버그 정보"):
                st.json({
                    "파일명": uploaded_files2.name,
                    "파일크기": f"{uploaded_files2.size:,} bytes",
                    "메시지 수": len(result.get("messages", []))
                })

    except Exception as e:
        st.error(f"❌ 오류 발생: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)




# 페이지 설정
st.set_page_config(
    page_title="고성군청 AI 도우미", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"  
)

st.markdown("""
    <style>
    /* 기본 바디 폰트 및 배경 */
    body {
        background-color: #f0f2f6;
        font-family: 'Noto Sans KR', sans-serif;
        color: #333;
    }

    
     /* 입력창 스타일 */
    .stChatInput input {
        width: 60%;  /* 입력창의 너비를 60%로 설정 */
        height: 10%;  /* 높이 증가 */    
        border: 2px solid #3b82f6;
        border-radius: 25px;
        padding: 15px 25px;
        font-size: 16px;
        background: linear-gradient(to right, #f0f9ff, #ffffff);
        transition: all 0.3s ease;
    }
    
    .stChatInput input:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        background: white;
    }
    
    .stChatInput button {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        border-radius: 50%;
        transition: transform 0.3s ease;
    }
    
    .stChatInput button:hover {
        transform: scale(1.1) rotate(15deg);
    }

    </style>
""", unsafe_allow_html=True)

animated_input_css = """
    <style>
    /* 입력창 등장 애니메이션 */
    .stChatInput {
        animation: slide-up 0.5s ease-out;
    }
    
    @keyframes slide-up {
        from {
            opacity: 0;
            transform: translateY(50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* 타이핑 효과 */
    .stChatInput input:focus {
        animation: typing-glow 2s ease-in-out infinite;
    }
    
    @keyframes typing-glow {
        0%, 100% { box-shadow: 0 0 5px rgba(59, 130, 246, 0.3); }
        50% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.6); }
    }
    
    /* 버튼 회전 효과 */
    .stChatInput button:hover {
        animation: rotate 0.5s ease;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    </style>
"""

st.markdown(animated_input_css, unsafe_allow_html=True)

# 타이틀
st.markdown("""
    <style>
        .centered-title {
            text-align: center;
            font-size: 3rem;
            color: #1e293b;
            margin-top: 0px;  /* 위쪽 마진 */
            margin-bottom: 3px;  /* 아래쪽 마진 */
            margin-left: 0px;  /* 왼쪽 마진 */
            margin-right: 0px;  /* 오른쪽 마진 */
        }
        .ai-text {
            font-size: 3.5rem; /* AI 글자 크기 */
            color: #2563eb;
            margin-left: 10px; /* AI 단어 왼쪽에 여백 추가 */
            margin-right: 10px; /* AI 단어 오른쪽 여백 추가 */
        }
    </style> 
    <h1 style="text-align: center; font-size: 3rem; color: #1e293b;">
    💬 고성군청 <span class="ai-text">AI</span> 도우미 </h>
                                
""", unsafe_allow_html=True)


# 사이드바 설정
with st.sidebar:
    st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
    
# 문서 학습기
    st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
    
    st.markdown("""
        <h2 style="text-align: center; font-size: 1.7rem; color: #000000;">📚 문서 학습기</h2>
        """, unsafe_allow_html=True)

    st.markdown("""
        <p class="upload-label">
            📎 PDF 파일 업로드 
            <span class="badge">최대 3개</span>
        </p>
    """, unsafe_allow_html=True)
    
    uploaded_files1 = st.file_uploader(
        "학습할 PDF 선택",
        type=['pdf'],
        accept_multiple_files=True,
        key="uploader1",
        label_visibility="collapsed"
    )
    
    # 업로드된 파일 표시
    if uploaded_files1:
        st.markdown("""
            <div style="background: #f0fdf4; padding: 0.5rem; border-radius: 8px; margin-top: 0.5rem;">
                <p style="margin: 0; font-size: 0.85rem; color: #15803d; font-weight: 500;">
                    ✅ {}개 파일 선택됨
                </p>
            </div>
        """.format(len(uploaded_files1)), unsafe_allow_html=True)
        
        for i, file in enumerate(uploaded_files1[:3], 1):
            st.markdown(f"""
                <div style="font-size: 0.8rem; color: #475569; padding: 0.2rem 0.5rem;">
                    {i}. {file.name[:30]}{'...' if len(file.name) > 30 else ''}
                </div>
            """, unsafe_allow_html=True)
    
    process1 = st.button(
        "🚀 학습 시작",
        key="process1",
        type="primary",
        # disabled=(uploaded_files1 is None or len(uploaded_files1) == 0),
        use_container_width=True
    )
    
    # 사용방법
    st.markdown("""
        <div class="usage-box">
            <div class="usage-title">
                💡 사용방법
            </div>
            <ol class="usage-list">
                <li>PDF 파일을 최대 3개까지 업로드</li>
                <li>"학습 시작" 버튼 클릭</li>
                <li>학습 완료 후 문서 기반 질문 가능</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 구분선
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    
    # ========== 섹션 2: PDF 분석기 ==========
    st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
    
    st.markdown("""
        <h2 style="text-align: center; font-size: 1.7rem; color: #000000;">🔍 PDF 분석기</h2>
        """, unsafe_allow_html=True)
    
   
    st.markdown("""
        <p class="upload-label">
            📎 분석할 PDF 업로드
            <span class="badge badge-blue">1개</span>
        </p>
    """, unsafe_allow_html=True)
    
    uploaded_files2 = st.file_uploader(
        "분석할 PDF 선택",
        type=['pdf'],
        key="uploader2",
        label_visibility="collapsed"
    )
    
    # 업로드된 파일 표시
    if uploaded_files2:
        st.markdown(f"""
            <div style="background: #eff6ff; padding: 0.5rem; border-radius: 8px; margin-top: 0.5rem;">
                <p style="margin: 0; font-size: 0.85rem; color: #1e40af; font-weight: 500;">
                    📄 {uploaded_files2.name[:35]}{'...' if len(uploaded_files2.name) > 35 else ''}
                </p>
                <p style="margin: 0.3rem 0 0 0; font-size: 0.75rem; color: #64748b;">
                    크기: {uploaded_files2.size / 1024:.1f} KB
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    process2 = st.button(
        "🚀 분석 시작",
        key="process2",
        type="primary",
        # disabled=(uploaded_files2 is None),
        use_container_width=True
    )
    
    # 사용방법
    st.markdown("""
        <div class="usage-box">
            <div class="usage-title">
                💡 사용방법
            </div>
            <ol class="usage-list">
                <li>PDF 파일 1개 업로드</li>
                <li>"분석 시작" 버튼 클릭</li>
                <li>키워드, 통계 등 분석 결과 확인</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 구분선
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    
    # 하단 정보 
    st.markdown("""
        <div style="text-align: center; padding: 1rem; color: #000000; font-size: 0.9rem;">
            <p style="margin: 0;">Made by 🔍 총무행정관 정보관리팀</p>
            <p style="margin: 0.5rem 0 0 0;">v1.0.0 | 2025</p>
        </div>
    """, unsafe_allow_html=True)

   

# 스트림릿 session_state에 메시지 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage("저는 고성군청 직원을 위해 최선을 다하는 인공지능 도우미입니다. "),  
        AIMessage("무엇을 도와 드릴까요?")
    ]

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

# 스트림릿 화면에 메시지 출력
for msg in st.session_state.messages:
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
if prompt := st.chat_input(placeholder="✨ 무엇이든 물어보세요?"):
    # 사용자 메시지 표시 및 저장
    st.chat_message("user").write(prompt)
    st.session_state.messages.append(HumanMessage(prompt))

    # vectorstore 존재 여부 확인
    vectorstore = st.session_state.get("vectorstore")
    
    if vectorstore is not None:
        # 벡터스토어 기반 답변
        with st.spinner("📚 학습된 문서를 검색하는 중..."):
            answer = answer_question(prompt)
        
        # 관련 문서가 없는 경우 일반 모드로 전환
        if answer and "죄송합니다. " in answer and len(answer) < 20:
            st.info("💡 학습된 문서에서 관련 내용을 찾지 못했습니다. 일반 AI 모드로 전환합니다.")
            response = get_ai_response(st.session_state["messages"])
            result = st.chat_message("assistant").write_stream(response)
            st.session_state["messages"].append(AIMessage(result))
        else:
            # 문서 기반 답변
            st.chat_message("assistant").write(answer)
            st.session_state.messages.append(AIMessage(answer))
    else:
        # 일반 AI 모드
        st.info("🤖 일반 AI 모드로 답변합니다. 문서를 학습하면 더 정확한 답변을 받을 수 있습니다.")
        response = get_ai_response(st.session_state["messages"])
        result = st.chat_message("assistant").write_stream(response)
        st.session_state["messages"].append(AIMessage(result))


# 문서 학습 함수 불러오기
if process1:
    st.session_state["vectorstore"] = process1_f(uploaded_files1)

# 문서 분석 함수 불러오기
if process2:
    process2_f(uploaded_files2)



