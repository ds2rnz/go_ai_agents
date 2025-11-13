import streamlit as st
import hashlib
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import os
from users_db import USERS_DB
import time
import subprocess

load_dotenv()

# USERS_DB = os.getenv("USERS_DB")


# 세션 로그 초기화
if "session_logs" not in st.session_state:
    st.session_state.session_logs = []

def dashboard_page():
    st.rerun()
    subprocess.run(["python", "test1.py"])

# 로그인 확인 함수
def check_login(user_id, name):
    if user_id in USERS_DB:
        user_info = USERS_DB[user_id]
        if user_info.get("name") == name:
            return True, user_info
    return False, None


st.set_page_config(page_title="로그인 시스템", page_icon="🔐", layout="wide")

# 세션 상태 초기화
if "usesr_id" not in st.session_state:
    st.session_state.user_id = False
if "login_id" not in st.session_state:
    st.session_state.login_id = False    
if "name" not in st.session_state:
    st.session_state.name = None
if "user_info" not in st.session_state:
    st.session_state.user_info = None
if "session_index" not in st.session_state:
    st.session_state.session_index = None

# 로그인 페이지
def login_page():
    st.title("🔐 로그인")
    
    # 중앙 정렬을 위한 컬럼
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("---")
        
        # 로그인 폼
        with st.form("login_form"):
            user_id = st.text_input("아이디", placeholder="새올아이디를 입력하세요")
            name = st.text_input("사용자이름",  placeholder="사용자 이름을 입력하세요")
            
            submit = st.form_submit_button("로그인", use_container_width=True)
            
            if submit:
                if user_id and name:
                    is_valid, user_info = check_login(user_id, name)
                    
                    if is_valid:
                        
                        st.success(f"환영합니다, {user_info['name']}님!")
                        time.sleep(3)
                        dashboard_page()
                        # st.rerun()  # 페이지 새로고침
                        # subprocess.Popen("streamlit run test1.py")

                    else:
                        st.error("아이디 또는 비밀번호가 올바르지 않습니다.")
                else:
                    st.warning("아이디와 비밀번호를 모두 입력해주세요.")
        
        st.markdown("---")
        
        # 테스트 계정 안내
        with st.expander("📝 사용자 계정 입력방법"):
            st.info("""
            **사용자 계정:**
            - 아이디: user12345  / 새올 로그인 ID 입력
            - 사용자이름: 홍길동  /  새올 ID 사용자명

            """)


login_page()

# CSS 스타일링
st.markdown("""
    <style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)
