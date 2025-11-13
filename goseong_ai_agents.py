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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from pathlib import Path
import tempfile
import traceback
import time
from users_db import USERS_DB


def check_login(user_id, name):
    if user_id in USERS_DB:
        user_info = USERS_DB[user_id]
        if user_info.get("name") == name:
            return True, user_info
    return False, None


def show_login_page():
    """로그인 페이지 표시"""

    st.title("🔐 로그인")
    
    # 중앙 정렬을 위한 컬럼
    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
  
        with st.form("login_form"):
            user_id = st.text_input("아이디", placeholder="새올아이디를 입력하세요")
            name = st.text_input("사용자이름",  placeholder="사용자 이름을 입력하세요")
            
            submit = st.form_submit_button("로그인", use_container_width=True)
            
            if submit:
                if user_id and name:
                    is_valid, user_info = check_login(user_id, name)
                    
                    if is_valid:
                        with st.spinner("로그인 중..."):
                            time.sleep(1)
                        
                        st.session_state.logged_in = True
                        st.session_state.user_info = user_info
                        st.success(f"환영합니다, {user_info['name']}님!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ 로그인 ID 또는 사용자이름을 올바르지 않습니다.")
                else:
                    st.warning("⚠️ 로그인 ID 또는 사용자이름을 모든 입력해주세요.")

        st.markdown('</div>', unsafe_allow_html=True)

        # 하단 정보
        st.markdown("""
            <div style="text-align: center; margin-top: 3rem; color: #64748b;">
                <p>Made by 🔍 총무행정관 정보관리팀</p>
                <p>v1.0.0 | 2025</p>
            </div>
        """, unsafe_allow_html=True)
       
        # 계정 안내
        with st.expander("📝 사용자 계정 입력방법"):
            st.info("""
            **사용자 계정:**
            - 아이디: user12345  / 새올 로그인 ID 입력
            - 사용자이름: 홍길동  /  새올 ID 사용자명

            """)



# ==================== 기존 함수들 ====================
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

@tool
def get_web_search(query: str) -> str:
    """
    웹 검색을 수행하는 함수.

    Args:
        query (str): 검색어
    Returns:
        str: 검색 결과
    """
    custom_wrapper = DuckDuckGoSearchAPIWrapper(region="kr-kr", time="y", max_results=10)
    search = DuckDuckGoSearchResults(
        api_wrapper=custom_wrapper,
        source="news, image, text",
        results_separator=';\n')
    
    results = search.run(query)

    st.toast("웹 검색을 통아여 알아보고 있습니다.", icon="🎉")
    return results

    

def load_vectorstore(embedding, persist_directory="C:/faiss_store"):
    
    # 저장 디렉토리가 존재하는지 확인
    if not os.path.isdir(persist_directory):
        return None
        
    index_file = os.path.join(persist_directory, "index.faiss")
    pkl_file = os.path.join(persist_directory, "index.pkl")
    

    if os.path.exists(index_file) and os.path.exists(pkl_file):
        try:
            st.spinner("📂 기존 학습한 자료를 불러오는 중...")
            vectorstore = FAISS.load_local(
                persist_directory, 
                embedding,
                allow_dangerous_deserialization=True
            )
            st.toast("기존 학습 데이터를 사용합니다!", icon="📚")
            return vectorstore
        except Exception as e:
            st.warning(f"⚠️ 기존 파일 로드 실패: {e}")
            return None
    else:
        return None        


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
            return "죄송합니다. 관련된 정보를 찾지 못했습니다."
        
        template = """당신은 친절한 AI 도우미입니다. 주어진 문서 내용을 바탕으로 질문에 답변해주세요.
    
                    문서 내용:
                    {context}

                    질문: {question}

                    답변 시 다음을 지켜주세요:
                    1. 문서 내용에 기반하여 정확하게 답변해주세요.
                    2. 가능한 한 구체적이고 자세하게 설명해주세요.
                    3. 한국어로 답변해주세요.

                    답변:"""

        prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
                )
        retriever = vectorstore.as_retriever(search_kwargs={"k":3})
        qa_chain = RetrievalQA.from_chain_type(
               llm=llm,
               chain_type="stuff",
               retriever=retriever,
               chain_type_kwargs={"prompt": prompt},
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
                

def ai_answer(messages):
    response = agent.invoke(
    {"messages": messages},
        config=config,
        tool_choice='any'
        )
    return response


def process1_f(uploaded_files1):
    """PDF 파일을 학습하여 벡터스토어 생성"""
    
    if uploaded_files1 and len(uploaded_files1) > 3:
        st.error("❌ PDF는 최대 3개까지 업로드 가능합니다!")
        st.warning("⚠️ PDF파일을 3개만 선택하여 주세요!")
        return None
    
    if not uploaded_files1:
        st.warning("⚠️ PDF 파일을 업로드해주세요.")
        return None

    try:
        with st.spinner("📚 PDF 임베딩 및 벡터스토어 생성 중... 잠시만 기다려주세요"):
            all_splits = []
            
            for idx, uploaded_file in enumerate(uploaded_files1, 1):
                st.write(f"📄 {idx}/{len(uploaded_files1)} 파일 처리 중: {uploaded_file.name}")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                try:
                    loader = PyPDFLoader(tmp_path)
                    data = loader.load()
                    
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=300, 
                        chunk_overlap=50
                    )
                    splits = splitter.split_documents(data)
                    all_splits.extend(splits)
                    
                    st.success(f"✅ {uploaded_file.name}: {len(splits)}개 문서로 분할")
                    
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            st.info(f"📊 총 문서 분할 수: {len(all_splits)}")

            embedding = OpenAIEmbeddings(
                model="text-embedding-3-large", 
                api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
            )
            
            persist_directory = "C:/faiss_store"
            try:
                os.makedirs(persist_directory, exist_ok=True)
            except Exception as e:
                st.error(f"❌ 디렉토리 생성 실패: {e}")
                return None

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
                        vectorstore = FAISS.from_documents(batch, embedding)
                    else:
                        vectorstore.add_documents(batch)
                    
                    vectorstore.save_local(persist_directory)
                    time.sleep(1.5)
                    
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


# ==================== 메인 앱 함수 ====================
def show_main_app():
    """메인 AI 도우미 앱"""
    
    # 페이지 설정
    st.markdown("""
        <style>
            .centered-title {
                text-align: center;
                font-size: 3rem;
                color: #1e293b;
                margin-top: 0px;
                margin-bottom: 3px;
            }
            .ai-text {
                font-size: 3.5rem;
                color: #2563eb;
                margin-left: 10px;
                margin-right: 10px;
            }
        </style> 
        <h1 style="text-align: center; font-size: 3rem; color: #1e293b;">
        💬 고성군청 <span class="ai-text">AI</span> 도우미 </h1>
    """, unsafe_allow_html=True)

    # 사이드바
    with st.sidebar:
        # 사용자 정보 표시
        st.markdown(f"""
            <div style="background: #e0f2fe; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <p style="margin: 0; font-weight: bold; color: #0c4a6e;">👤 {st.session_state.user_info['name']}</p>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem; color: #075985;">ID: {st.session_state.user_info['login_id']}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # 로그아웃 버튼
        if st.button("🚪 로그아웃", type="secondary", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_info = None
            st.rerun()
        
        st.markdown("---")
        
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
            use_container_width=True
        )
        
        st.markdown("---")
        st.markdown("### 📖 :blue[사용방법]")
        st.markdown("""
            1. PDF 파일을 업로드하세요(최대 3개만)
            2. "학습시작" 버튼을 클릭하세요
            3. 학습한 문서를 바탕으로 사용자 요청에 따라 답변합니다.
            """)
            
        st.markdown("---")

        st.markdown("""
            <div style="text-align: center; padding: 1rem; color: #000000; font-size: 0.9rem;">
                <p style="margin: 0;">Made by 🔍 총무행정관 정보관리팀</p>
                <p style="margin: 0.5rem 0 0 0;">v1.0.0 | 2025</p>
            </div>
        """, unsafe_allow_html=True)

    # 메시지 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "저는 고성군청 직원을 위해 최선을 다하는 인공지능 도우미입니다."},
            {"role": "assistant", "content": "무엇이을 도와 드릴까요?"}
        ]

    # 메시지 출력
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        st.chat_message(role).write(content)

    # vectorstore 로드
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = load_vectorstore(
            embedding=embedding,
            persist_directory="C:/faiss_store"
        )

    # 사용자 입력 처리
    if prompt := st.chat_input(placeholder="무엇이든 물어보세요?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        vectorstore = st.session_state.get("vectorstore")

        if vectorstore is not None:
            with st.spinner("📚 학습된 문서를 검색하는 중..."):
                answer = answer_question(prompt)

            if answer and "죄송합니다. " in answer or len(answer) < 30:
                st.info("💡 학습된 문서에서 관련 내용을 찾지 못했습니다. 일반 AI 모드로 전환합니다.")
                
                with st.spinner("답변 생성 중..."):
                    try:
                        response = ai_answer(st.session_state.messages)
                        ai_response = response['messages'][-1].content
                        st.toast("일반 AI 모드로 답변합니다....!", icon="🎉")
                        
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})
                        st.chat_message("assistant").write(ai_response)
                    except Exception as e:
                        error_msg = f"오류가 발생했습니다: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        st.chat_message("assistant").write(error_msg)
            else:
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.chat_message("assistant").write(answer)
        else:
            with st.spinner("답변 생성 중..."):
                try:
                    response = ai_answer(st.session_state.messages)
                    ai_response = response['messages'][-1].content
                    st.toast("일반 AI 모드로 답변합니다....!", icon="🎉")
                    
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    st.chat_message("assistant").write(ai_response)
                except Exception as e:
                    error_msg = f"오류가 발생했습니다: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.chat_message("assistant").write(error_msg)

    # 문서 학습 처리
    if process1:
        st.session_state["vectorstore"] = process1_f(uploaded_files1)


# ==================== 메인 실행 ====================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

config = {"configurable": {"thread_id": "1"}}

llm = init_chat_model(
    model = "openai:gpt-4o",
    temperature=0.6, 
    max_tokens=1000, 
    timeout=10, 
    max_retries=2, 
    )

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large", 
    api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    )

agent = create_agent(
    model=llm,
    tools=[get_current_time, get_web_search],
    middleware=[],
    system_prompt="사용자가 질문을하면 구체적이고 자세하게 설명해주고 모르는 내용이면 인터넷 검색을 꼭해서 답변해줘 그리고 한글로 답해주세요", 
    )

# 페이지 설정
st.set_page_config(page_title="GPT 기반 AI 도우미", page_icon="💬", layout="wide")

# 세션 상태 초기화
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = None

# 로그인 상태에 따라 페이지 표시
if not st.session_state.logged_in:
    show_login_page()
else:
    show_main_app()
