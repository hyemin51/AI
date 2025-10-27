# rag_pipeline.py
import os
from dotenv import load_dotenv

# -------------------------------------------
# 0. 환경변수 로드 (.env 또는 Streamlit Secrets)
# -------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------------------------------
# 1. LangChain 관련 import
# -------------------------------------------
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# -------------------------------------------
# 2. 문서 로드 함수
# -------------------------------------------
def load_docs(docs_path="docs"):
    """docs 폴더 안의 pdf/txt 파일을 모두 읽어서 LangChain 문서 리스트로 반환"""
    docs = []

    # PDF 파일
    pdf_loader = DirectoryLoader(docs_path, glob="*.pdf", loader_cls=PyPDFLoader)
    docs.extend(pdf_loader.load())

    # TXT 파일
    txt_loader = DirectoryLoader(docs_path, glob="*.txt", loader_cls=TextLoader)
    docs.extend(txt_loader.load())

    return docs


# -------------------------------------------
# 3. 문서 청크 분할
# -------------------------------------------
def split_docs(documents, chunk_size=800, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(documents)


# -------------------------------------------
# 4. 벡터스토어 생성 및 저장
# -------------------------------------------
def build_vectorstore(chunks, save_path="vectorstore"):
    """문서 청크 → 임베딩 → FAISS 벡터DB 저장"""
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)
    vectordb.save_local(save_path)
    return vectordb


# -------------------------------------------
# 5. 기존 벡터스토어 불러오기
# -------------------------------------------
def load_vectorstore(save_path="vectorstore"):
    """이미 만들어둔 벡터DB를 다시 로드"""
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectordb = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectordb


# -------------------------------------------
# 6. 질문 → 답변 함수 생성 (프롬프트 포함)
# -------------------------------------------
def make_answer_function(vectordb):
    """
    make_answer_function(...) -> answer_question(question: str) 형태의 함수를 리턴.
    Streamlit에서 바로 호출 가능.
    """

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # 🧠 프롬프트 (LLM에게 줄 지시문)
    prompt = ChatPromptTemplate.from_template(
        """너는 대학생 수준의 회계/재무 튜터 챗봇이야.
아래의 참고 문서를 바탕으로 질문에 답변해줘.
문서(context)에 없는 내용은 "그 부분은 자료에 없습니다"라고 정직하게 말해.
절대 추측하거나 지어내지 마.

[참고 문서]
{context}

[사용자 질문]
{question}

위 내용을 바탕으로 한국어로 친절하고 명확하게 설명해줘."""
    )

    def answer_question(user_question: str) -> str:
        # (1) 문서 검색
        docs = retriever.get_relevant_documents(user_question)
        context_text = "\n\n".join([d.page_content for d in docs])

        # (2) LLM 초기화
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-4o-mini",
            temperature=0.2,
        )

        # (3) 프롬프트 채우기
        filled_prompt = prompt.format(context=context_text, question=user_question)

        # (4) LLM 호출
        response = llm.invoke(filled_prompt)

        # (5) 텍스트만 반환
        return getattr(response, "content", str(response))

    return answer_question

