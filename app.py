from langchain_community.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
import os

# 프롬프트 템플릿 정의
answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

# 웹사이트 파싱 함수
def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )

# Cloudflare AI 문서 로드 및 색인 생성 함수
@st.cache_data(show_spinner="Cloudflare AI 문서 로드 중...")
def load_cloudflare_docs(api_key):
    """
    Cloudflare AI 제품군(AI Gateway, Vectorize, Workers AI)의 문서를 로드하고
    FAISS 벡터 저장소로 변환하여 리트리버를 반환합니다.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    
    # Cloudflare 개발자 문서 sitemap 및 AI 제품 필터
    sitemap_url = "https://developers.cloudflare.com/sitemap.xml"
    filter_urls = [
        "https://developers.cloudflare.com/ai-gateway/",
        "https://developers.cloudflare.com/vectorize/",
        "https://developers.cloudflare.com/workers-ai/",
    ]

    loader = SitemapLoader(
        sitemap_url,
        filter_urls=filter_urls,
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    
    docs = loader.load_and_split(text_splitter=splitter)
    
    # OpenAI 임베딩 및 FAISS 벡터 저장소 생성 (API 키 사용)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store.as_retriever()

# Streamlit 앱 설정
st.set_page_config(
    page_title="Cloudflare AI SiteGPT",
    page_icon="☁️",
)

st.markdown(
    """
    # ☁️ Cloudflare AI SiteGPT
            
    AI Gateway, Vectorize, Workers AI에 대해 무엇이든 물어보세요.
    
    **시작하려면 사이드바에 OpenAI API 키를 입력하세요.**
"""
)

# 사이드바 설정
with st.sidebar:
    # OpenAI API 키 입력
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
    )
    
    # 깃허브 링크
    st.markdown(
        "--- \n"
        "[View on GitHub](https://github.com/jj-prog3/streamlit-site)"
    )

# API 키가 입력되었을 때만 앱 로직 실행
if api_key:
    # 1. API 키를 사용하여 LLM 모델 초기화
    llm = ChatOpenAI(
        temperature=0.1,
        openai_api_key=api_key
    )

    # 2. LLM을 사용하는 답변 생성 함수 정의
    def get_answers(inputs):
        docs = inputs["docs"]
        question = inputs["question"]
        answers_chain = answers_prompt | llm
        
        return {
            "question": question,
            "answers": [
                {
                    "answer": answers_chain.invoke(
                        {"question": question, "context": doc.page_content}
                    ).content,
                    "source": doc.metadata["source"],
                    "date": doc.metadata.get("lastmod", "N/A"), # 'lastmod'가 없을 경우 대비
                }
                for doc in docs
            ],
        }

    # 3. LLM을 사용하는 최종 답변 선택 함수 정의
    def choose_answer(inputs):
        answers = inputs["answers"]
        question = inputs["question"]
        choose_chain = choose_prompt | llm
        
        condensed = "\n\n".join(
            f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
            for answer in answers
        )
        
        return choose_chain.invoke(
            {
                "question": question,
                "answers": condensed,
            }
        )

    # 4. 문서 로드 (캐시됨)
    retriever = load_cloudflare_docs(api_key)

    # 5. 사용자 질문 입력
    query = st.text_input("Cloudflare AI 문서에 대해 질문하세요:")

    if query:
        # 6. LangChain 체인 설정
        chain = (
            {
                "docs": retriever,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )
        
        # 7. 체인 실행 및 결과 표시
        with st.spinner("답변을 생성 중입니다..."):
            result = chain.invoke(query)
            st.markdown(result.content.replace("$", "\$"))

# API 키가 입력되지 않았을 경우 안내 메시지
else:
    st.info("시작하려면 사이드바에 OpenAI API 키를 입력하세요.")
