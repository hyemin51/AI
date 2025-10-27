# rag_pipeline.py
import os
from dotenv import load_dotenv

# .env 읽기
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LangChain 최신 구조
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# .env 안에 있는 OPENAI_API_KEY 불러오기
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


### 1) 문서 불러오기
def load_docs(docs_path="docs"):
    """
    docs 폴더 안의 pdf / txt 파일들을 전부 읽어서 LangChain 문서 리스트로 반환
    """
    docs = []

    # PDF 로더 (docs/*.pdf)
    pdf_loader = DirectoryLoader(
        docs_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    docs.extend(pdf_loader.load())

    # TXT 로더 (docs/*.txt)
    txt_loader = DirectoryLoader(
        docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )
    docs.extend(txt_loader.load())

    return docs


### 2) 문서 쪼개기 (chunking)
def split_docs(documents, chunk_size=800, chunk_overlap=150):
    """
    긴 문서를 LLM이 다룰 수 있게 작은 청크로 분할
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)


### 3) 벡터스토어 만들고 저장
def build_vectorstore(chunks, save_path="vectorstore"):
    """
    문서 청크 -> 임베딩 -> FAISS 벡터DB 생성
    그리고 vectorstore/ 폴더에 저장
    """
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)
    vectordb.save_local(save_path)
    return vectordb


### 4) 기존 벡터스토어 불러오기
def load_vectorstore(save_path="vectorstore"):
    """
    이미 만들어둔 벡터DB를 다시 로드
    """
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectordb = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True  # FAISS 로딩할 때 필요
    )
    return vectordb


### 5) 질문→검색→답변 파이프라인(RAG 체인) 만들기
def make_qa_chain(vectordb):
    """
    사용자의 질문이 들어오면
    1) vectordb에서 관련 청크 검색하고
    2) 그 청크랑 질문을 llm한테 주고
    3) 답변 받아오는 체인 생성
    """
    retriever = vectordb.as_retriever(search_k=4)

    # 모델에게 줄 기본 지침 프롬프트
    prompt = ChatPromptTemplate.from_template("""
너는 수업 과제용 RAG 챗봇이야.
아래의 context(문서 내용)를 기반으로만 한국어로 대답해.
모르면 모른다고 말해. 절대 지어내지 마.

[context]
{context}

[question]
{question}
""")

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",  # 수업에서 허용된 모델/플랜에 맞춰 조정 가능
        temperature=0.2       # 너무 창의적으로 헛소리 안 하게 낮게
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",          # 가장 단순한 RAG 방식
        retriever=retriever,
        chain_type_kwargs={
            "prompt": prompt
        },
        return_source_documents=True
    )

    return qa_chain

from rag_pipeline import load_docs, split_docs, build_vectorstore

docs = load_docs("docs")          # docs 폴더에 넣어둔 pdf/txt 읽기
chunks = split_docs(docs)         # 문서를 작은 청크로 분리
build_vectorstore(chunks)         # vectorstore/ 폴더에 FAISS 인덱스 저장
def make_qa_chain(vectorstore):
    """
    vectorstore에서 비슷한 chunk를 찾아오고,
    그 chunk들을 context로 해서 LLM(ChatOpenAI)에게 답변을 생성시키는 간단한 QA 체인.
    최신 LangChain의 RetrievalQA 클래스 없이 직접 만듦.
    """

    # 1) 벡터스토어에서 Retriever 만들기
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2) 프롬프트 템플릿 만들기
    prompt = ChatPromptTemplate.from_template(
        """너는 보안/규제/금융 데이터에 대한 어시스턴트야.
아래는 참고할 문서 내용이야:

{context}

사용자 질문:
{question}

문서에서 근거를 사용해서, 한국어로 명확하고 간단하게 답해줘.
가능하면 근거에 없는 내용을 막 추측하지 말고 없다고 말해줘."""
    )

    # 3) QA 함수를 만들어서 리턴
    #    이 함수는 사용자의 질문(q)을 받아서 답변 텍스트를 돌려줄 거야.
    def answer_question(q):
        # (a) 관련 context 뽑기
        docs = retriever.get_relevant_documents(q)
        context_text = "\n\n".join([d.page_content for d in docs])

        # (b) LLM 호출 준비
        llm = ChatOpenAI(
            model="gpt-4o-mini",  # 네가 쓰고 싶은 모델명 (필요에 맞게 바꿔도 됨)
            temperature=0.2,
        )

        # (c) 프롬프트 채우기
        filled_prompt = prompt.format(
            context=context_text,
            question=q
        )

        # (d) 실제 LLM에게 답변 받기
        response = llm.invoke(filled_prompt)

        # response는 메시지 객체일 가능성이 높음 -> content만 추출
        return getattr(response, "content", str(response))

    return answer_question
