import os
import random
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# RAG 관련 함수 불러오기
from rag_pipeline import (
    load_docs,
    split_docs,
    build_vectorstore,
    load_vectorstore,
    make_answer_function,
)

# --------------------------------------------------
# 환경 변수 로드
# --------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --------------------------------------------------
# Streamlit 기본 설정
# --------------------------------------------------
st.set_page_config(page_title="나만의 회계 튜터", page_icon="📚", layout="wide")

st.title("📚 나만의 회계 RAG + 퀴즈 챗봇")
st.write("이 앱은 GitHub 리포지토리에 포함된 자료만으로 동작해요. 로컬 PC 경로에 의존하지 않아요 👍")
st.write("• 문제 은행: `data/accounting_bank_full.csv`\n• 난이도별 랜덤 출제 가능 (easy / medium / hard / 전체)\n• 아래 입력창에 회계 질문을 적으면 실제 문서 기반 답변을 생성해줘요 ✨")

st.markdown("---")

# --------------------------------------------------
# 세션 초기화
# --------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []
if "answer_fn" not in st.session_state:
    st.session_state["answer_fn"] = None
if "vector_ready" not in st.session_state:
    st.session_state["vector_ready"] = False

# --------------------------------------------------
# 벡터스토어 로드 or 생성
# --------------------------------------------------
if not st.session_state["vector_ready"]:
    try:
        vectordb = load_vectorstore("vectorstore")
    except Exception:
        try:
            docs = load_docs("docs")
            chunks = split_docs(docs)
            vectordb = build_vectorstore(chunks, save_path="vectorstore")
        except Exception as e:
            vectordb = None
            st.error("❌ 벡터스토어를 불러오거나 생성하지 못했어요. docs 폴더와 API 키를 확인해주세요.")
            st.code(str(e))

    if vectordb is not None:
        st.session_state["answer_fn"] = make_answer_function(vectordb)
        st.session_state["vector_ready"] = True

# --------------------------------------------------
# 회계 질문 영역
# --------------------------------------------------
st.markdown("## 💬 회계 질문해 보세요")

user_q = st.text_input(
    "예: '자산이 뭐예요?', '발생주의 회계를 쉽게 설명해줘', '선급비용은 왜 자산이에요?' 등",
    key="question_input_cloudonly",
)

ask_button = st.button("질문하기")

if ask_button:
    if not user_q.strip():
        st.warning("질문을 입력해 주세요.")
    elif not st.session_state["vector_ready"]:
        st.error("RAG 검색 기능이 초기화되지 않았어요 (벡터스토어 준비 실패).")
    else:
        with st.spinner("답변 생성 중..."):
            answer_text = st.session_state["answer_fn"](user_q)

        st.session_state["history"].append({"role": "user", "content": user_q})
        st.session_state["history"].append({"role": "assistant", "content": answer_text})

        st.markdown("#### 📌 답변")
        st.write(answer_text)

st.markdown("---")

# --------------------------------------------------
# 회계 퀴즈 섹션
# --------------------------------------------------
st.markdown("## 📝 회계원리 퀴즈")

CSV_PATH = "data/accounting_bank_full.csv"

@st.cache_data
def load_question_bank(csv_path: str):
    df = pd.read_csv(csv_path)
    needed_cols = [
        "week",
        "topic",
        "question",
        "choices",
        "answer",
        "explanation",
        "difficulty",
    ]
    return df[needed_cols]

bank_df = None
load_error = None
try:
    bank_df = load_question_bank(CSV_PATH)
except Exception as e:
    load_error = str(e)

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### 🔎 랜덤 문제 받기")
    quiz_btn = st.button("문제 출제")

with col_right:
    difficulty_choice = st.selectbox(
        "난이도 선택:",
        ["전체", "easy", "medium", "hard"],
        index=0,
    )

if bank_df is None:
    st.error("❌ 퀴즈 CSV를 불러오지 못했어요. (data/accounting_bank_full.csv)")
    if load_error:
        st.code(load_error, language="text")
else:
    if quiz_btn:
        pool_df = bank_df if difficulty_choice == "전체" else bank_df[bank_df["difficulty"] == difficulty_choice]
        if len(pool_df) == 0:
            st.warning(f"'{difficulty_choice}' 난이도 문제를 찾을 수 없어요.")
        else:
            row = pool_df.sample(1).iloc[0]
            st.markdown(f"**📚 주차:** {row['week']}주차 / **주제:** {row['topic']}")
            st.markdown("**❓ 문제**")
            st.write(row['question'])
            if isinstance(row['choices'], str) and row['choices'].strip():
                st.markdown("**보기**")
                for choice in row['choices'].split("|"):
                    st.write("- " + choice.strip())
            with st.expander("✅ 정답 보기 / 해설 보기"):
                st.markdown("**정답:**")
                st.write(row['answer'])
                st.markdown("**해설:**")
                st.write(row['explanation'])

st.markdown("---")

# --------------------------------------------------
# 대화 기록
# --------------------------------------------------
st.markdown("## 💬 대화 기록")

if len(st.session_state["history"]) == 0:
    st.write("아직 대화가 없어요 🙇")
else:
    for turn in st.session_state["history"]:
        if turn["role"] == "user":
            st.markdown(f"**🙋 사용자:** {turn['content']}")
        else:
            st.markdown(f"**🤖 챗봇:** {turn['content']}")
