from langchain_community.document_loaders import SitemapLoader
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
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

    filtered_docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]
    
    if not filtered_docs:
        # 필터링 후 문서가 하나도 없으면 오류를 발생시키고 앱을 중지합니다.
        st.error("문서를 로드하거나 파싱할 수 없습니다. 사이트맵 URL을 확인하거나 나중에 다시 시도하세요.")
        st.stop()


    
    # OpenAI 임베딩 및 FAISS 벡터 저장소 생성 (API 키 사용)
    
    # --- 오류 수정 (BadRequestError - 토큰 한도 초과) ---
    # chunk_size를 설정하여 단일 API 요청의 토큰 수가 한도를 넘지 않도록 
    # 임베딩 요청을 더 작은 배치로 나눕니다. (기본값 2048이 너무 컸음)
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
        chunk_size=1000  # 1000개 문서 단위로 나누어 API 요청
    )
    # --- 오류 수정 완료 ---
    
    # 필터링된 'filtered_docs' 리스트를 사용하여 벡터 저장소를 생성합니다.
    vector_store = FAISS.from_documents(filtered_docs, embeddings) 
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
    try:
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

    except Exception as e:
        # load_cloudflare_docs에서 st.stop()이 호출되면 이 코드는 실행되지 않을 수 있지만,
        # 임베딩 생성 중 다른 예외 발생 시를 대비해 처리합니다.
        st.error(f"문서를 로드하거나 처리하는 중 오류가 발생했습니다: {e}")
        st.warning("API 키가 유효한지, 크레딧이 남아있는지 확인하세요.")


# API 키가 입력되지 않았을 경우 안내 메시지
else:
    st.info("시작하려면 사이드바에 OpenAI API 키를 입력하세요.")

