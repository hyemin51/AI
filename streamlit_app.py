import random
import pandas as pd
import streamlit as st

# =========================================
# 스트림릿 기본 설정
# =========================================
st.set_page_config(
    page_title="나만의 회계 튜터",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .question-box textarea {
        font-size: 1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 나만의 회계 RAG + 퀴즈 챗봇")
st.write(
    "이 앱은 GitHub 리포지토리 안에 있는 자료만 사용해서 동작해요. "
    "로컬 PC 경로(C:\\... 이런 거)에 의존하지 않아요. 👍"
)
st.write(
    "• 문제 은행: `data/accounting_bank_full.csv`\n"
    "• 난이도별 랜덤 출제 가능 (easy / medium / hard / 전체)\n"
    "• 아래 입력창에 회계 질문을 적으면 기록만 남겨줘요 (지금은 자동 답변은 임시 메시지)"
)

st.markdown("---")

# =========================================
# 1. 회계 질문 섹션
# =========================================

st.markdown("## 💬 회계 질문해 보세요")

# 세션 스토리지에 대화 히스토리 없으면 만들기
if "history" not in st.session_state:
    st.session_state["history"] = []

user_q = st.text_input(
    "예: '자산이 뭐예요?', '발생주의 회계 쉽게 설명해줘', '선급비용은 왜 자산이에요?' 등",
    key="question_input_cloudonly",
)

ask_button = st.button("질문하기")

if ask_button:
    if not user_q.strip():
        st.warning("질문을 입력해 주세요.")
    else:
        # 아직은 LLM 호출 안 하고 안내 문구만 답변으로 넣음
        answer_text = (
            "지금은 Streamlit Cloud 전용 버전이라 자동 설명은 준비 중이에요. "
            "질문은 아래 '대화 기록'에 저장해 둘게요 🙂"
        )

        # 히스토리에 user / bot 턴 둘 다 추가
        st.session_state["history"].append({"role": "user", "content": user_q})
        st.session_state["history"].append({"role": "assistant", "content": answer_text})

        st.markdown("#### 📌 답변(임시)")
        st.write(answer_text)

st.markdown("---")

# =========================================
# 2. 회계원리 퀴즈 섹션
# =========================================

st.markdown("## 📝 회계원리 퀴즈")

CSV_PATH = "data/accounting_bank_full.csv"

@st.cache_data
def load_question_bank(csv_path: str):
    """
    data/accounting_bank_full.csv 를 읽어서 DataFrame으로 반환.
    GitHub/Streamlit Cloud 기준 경로로 동작.
    """
    df = pd.read_csv(csv_path)

    # 안전하게 필요한 컬럼만 남겨서 돌려주자.
    needed_cols = [
        "week",
        "topic",
        "question",
        "choices",
        "answer",
        "explanation",
        "difficulty",
    ]
    df = df[needed_cols]
    return df

# CSV 읽기 시도
bank_df = None
load_error = None
try:
    bank_df = load_question_bank(CSV_PATH)
except Exception as e:
    load_error = str(e)

# 컬럼 2개로 나눠서 왼쪽은 버튼, 오른쪽은 난이도 선택
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### 🔎 랜덤 문제 받기")
    quiz_btn = st.button("문제 출제")

with col_right:
    difficulty_choice = st.selectbox(
        "난이도 선택(선택 시 난이도에 맞춰 랜덤):",
        ["전체", "easy", "medium", "hard"],
        index=0,
    )

# 만약 CSV 로드 실패했다면 에러 메시지
if bank_df is None:
    st.error("❌ 퀴즈 CSV를 불러오지 못했어요. (data/accounting_bank_full.csv)")
    if load_error:
        st.code(load_error, language="text")
else:
    # 정상적으로 불러왔다면 버튼 클릭 시 문제 뽑기
    if quiz_btn:
        # 난이도 필터링
        if difficulty_choice == "전체":
            pool_df = bank_df
        else:
            pool_df = bank_df[bank_df["difficulty"] == difficulty_choice]

        if len(pool_df) == 0:
            st.warning(f"'{difficulty_choice}' 난이도 문제를 찾을 수 없어요.")
        else:
            # 랜덤 1문제
            row = pool_df.sample(1).iloc[0]

            week = row.get("week", "N/A")
            topic = row.get("topic", "")
            question = row.get("question", "")
            choices_raw = str(row.get("choices", ""))
            answer = row.get("answer", "")
            explanation = row.get("explanation", "")

            st.markdown(f"**📚 주차:** {week}주차 / **주제:** {topic}")
            st.markdown("**❓ 문제**")
            st.write(question)

            # 보기 출력
            if isinstance(choices_raw, str) and choices_raw.strip() not in ["", "nan", "None"]:
                st.markdown("**보기**")
                for choice in choices_raw.split("|"):
                    st.write("- " + choice.strip())

            # 정답 / 해설
            with st.expander("✅ 정답 보기 / 해설 보기"):
                st.markdown("**정답:**")
                st.write(answer)
                st.markdown("**해설:**")
                st.write(explanation)

st.markdown("---")

# =========================================
# 3. 대화 기록 섹션
# =========================================

st.markdown("## 💬 대화 기록")

if len(st.session_state["history"]) == 0:
    st.write("아직 대화가 없어요 🙇")
else:
    for turn in st.session_state["history"]:
        if turn["role"] == "user":
            st.markdown(f"**🙋 사용자:** {turn['content']}")
        else:
            st.markdown(f"**🤖 챗봇:** {turn['content']}")
