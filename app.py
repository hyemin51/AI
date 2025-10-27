import sys, os
import random
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# LangChain / OpenAI 관련
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ----------------------------------------------------------------
# 환경 변수 로드 (Streamlit Secrets 또는 .env)
# ----------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ----------------------------------------------------------------
# RAG 유틸 함수들 (원래 rag_pipeline.py 안에 있던 내용)
# ----------------------------------------------------------------

def load_docs(docs_path="docs"):
    """
    docs 폴더 안의 pdf / txt 파일을 모두 읽어서 LangChain 문서들(list[Document])로 반환
    """
    docs = []

    # PDF 로더
    pdf_loader = DirectoryLoader(
        docs_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    docs.extend(pdf_loader.load())

    # TXT 로더
    txt_loader = DirectoryLoader(
        docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
    )
    docs.extend(txt_loader.load())

    return docs


def split_docs(documents, chunk_size=800, chunk_overlap=150):
    """
    긴 문서를 LLM이 다룰 수 있게 작은 청크로 분할
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks, save_path="vectorstore"):
    """
    분할된 문서(chunks)를 OpenAI 임베딩으로 벡터화하고,
    FAISS 벡터스토어로 만들고 디스크에 저장
    """
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)
    vectordb.save_local(save_path)
    return vectordb


def load_vectorstore(save_path="vectorstore"):
    """
    이미 만들어둔 FAISS 벡터스토어를 디스크에서 다시 불러오기
    """
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectordb = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True,  # FAISS에서 필요
    )
    return vectordb


def make_answer_function(vectordb):
    """
    벡터스토어를 받아서, 사용자의 질문 q -> 답변 텍스트 를 돌려주는 answer_fn 을 만들어 돌려줌
    (RetrievalQA 비슷하게 직접 구성)
    """

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        """너는 회계 과목 보조 강사야.
아래는 참고할 문서 내용이야:

{context}

사용자 질문:
{question}

문서에서 근거를 사용해서 한국어로 정확하고 쉽게 설명해줘.
모르면 모른다고 말해. 근거 없는 내용은 지어내지 마."""
    )

    def answer_fn(user_question: str) -> str:
        # 1) 관련 청크 검색
        docs = retriever.get_relevant_documents(user_question)
        context_text = "\n\n".join([d.page_content for d in docs])

        # 2) LLM 호출 준비
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-4o-mini",
            temperature=0.2,
        )

        # 3) 프롬프트 채우기
        filled_prompt = prompt.format(
            context=context_text,
            question=user_question,
        )

        # 4) 실제 모델 호출
        response = llm.invoke(filled_prompt)

        # response는 메시지 객체일 수 있으므로 content 속성을 우선 사용
        return getattr(response, "content", str(response))

    return answer_fn


# ----------------------------------------------------------------
# Streamlit UI 시작
# ----------------------------------------------------------------

st.set_page_config(
    page_title="나만의 회계 튜터",
    page_icon="📚",
    layout="wide"
)

st.title("📚 나만의 회계 RAG + 퀴즈 챗봇")
st.write(
    "이 앱은 GitHub 리포지토리에 포함된 자료만으로 동작해요. "
    "로컬 PC 경로에 의존하지 않아요 👍"
)
st.write(
    "• 문제 은행: `data/accounting_bank_full.csv`\n"
    "• 난이도별 랜덤 출제 가능 (easy / medium / hard / 전체)\n"
    "• 아래 입력창에 회계 질문을 적으면 실제 문서 기반 답변을 생성해줘요 ✨"
)

st.markdown("---")

# 세션 상태 초기화
if "history" not in st.session_state:
    st.session_state["history"] = []

if "answer_fn" not in st.session_state:
    st.session_state["answer_fn"] = None

if "vector_ready" not in st.session_state:
    st.session_state["vector_ready"] = False

# --------------------------------------------------
# 벡터스토어 로드 or 새로 구축
# --------------------------------------------------
if not st.session_state["vector_ready"]:
    try:
        vectordb = load_vectorstore("vectorstore")
    except Exception:
        try:
            docs = load_docs("docs")      # docs/ 안 PDF, TXT
            chunks = split_docs(docs)     # 청크 나누기
            vectordb = build_vectorstore(
                chunks,
                save_path="vectorstore"
            )
        except Exception as e:
            vectordb = None
            st.error(
                "❌ 벡터스토어를 불러오거나 생성하지 못했어요. "
                "docs 폴더와 OPENAI_API_KEY를 확인해주세요."
            )
            st.code(str(e), language="text")

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

left_col, right_col = st.columns(2)

with left_col:
    st.markdown("#### 🔎 랜덤 문제 받기")
    quiz_btn = st.button("문제 출제")

with right_col:
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
        # 난이도별 문제 풀에서 하나 뽑기
        if difficulty_choice == "전체":
            pool_df = bank_df
        else:
            pool_df = bank_df[bank_df["difficulty"] == difficulty_choice]

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
# 대화 기록 섹션
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

