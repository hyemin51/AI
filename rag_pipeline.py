def make_answer_function(vectordb):
    """
    vectorstore에서 비슷한 chunk를 찾아오고,
    그 chunk들을 context로 해서 LLM(ChatOpenAI)에게 답변을 생성하는 함수 생성.
    """
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    def answer_question(user_question: str) -> str:
        # 1) 유사 문서 검색
        # LangChain 최신 retriever는 .invoke(query) 형태로 호출 가능
        retrieved_docs = retriever.invoke(user_question)

        # 일부 버전에서는 invoke()가 단일 Document 또는 list[Document]를 줄 수 있어서
        # 리스트 형태로 정규화
        if not isinstance(retrieved_docs, list):
            retrieved_docs = [retrieved_docs]

        context_text = "\n\n".join(
            [doc.page_content for doc in retrieved_docs if hasattr(doc, "page_content")]
        )

        # 2) LLM 준비
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-4o-mini",
            temperature=0.2,
        )

        # 3) 프롬프트 구성 - context랑 질문을 하나의 메시지로 전달
        system_prompt = f"""
너는 대학생 수준의 회계/재무 튜터 챗봇이야.
아래 제공된 참고 문서(context)를 기반으로만 한국어로 답해.
모르면 모른다고 말해. 억지로 지어내지 마.

[context]
{context_text}

[question]
{user_question}

위 내용을 근거로 명확하고 쉽게 설명해줘.
가능하면 회계 기초 개념부터 차근차근 설명해.
"""

        # 4) LLM 호출
        response = llm.invoke(system_prompt)

        # 5) 답변 텍스트만 추출
        return getattr(response, "content", str(response))

    return answer_question

