from langchain_community.document_loaders import SitemapLoader
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import os

# --- 캐시 구현 ---
# 1. 세션 상태에 쿼리 캐시 및 API 키 저장소 초기화
if "query_cache" not in st.session_state:
    st.session_state.query_cache = {}
if "cached_api_key" not in st.session_state:
    st.session_state.cached_api_key = None
# --- 캐시 구현 완료 ---


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
@st.cache_resource(show_spinner="Cloudflare AI 문서 로드 중...")
def load_cloudflare_docs(api_key):
    """
    Cloudflare AI 제품군(AI Gateway, Vectorize, Workers AI)의 문서를 로드하고
    FAISS 벡터 저장소로 변환하여 리트리버를 반환합니다.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder( 
        chunk_size=1000,
        chunk_overlap=200,
    )
    
    # --- 오류 수정 (Invalid URL) ---
    sitemap_url = "https://developers.cloudflare.com/sitemap.xml"
    filter_urls = [
        "https://developers.cloudflare.com/ai-gateway/",
        "https://developers.cloudflare.com/vectorize/",
        "https://developers.cloudflare.com/workers-ai/",
    ]
    # --- 오류 수정 완료 ---

    loader = SitemapLoader(
        sitemap_url,
        filter_urls=filter_urls,
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    
    docs = loader.load_and_split(text_splitter=splitter)

    # 빈 문서 필터링 (BadRequestError 방지)
    filtered_docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]
    
    if not filtered_docs:
        st.error("문서를 로드하거나 파싱할 수 없습니다. 사이트맵 URL을 확인하거나 나중에 다시 시도하세요.")
        st.stop()
    
    # OpenAI 임베딩 설정 (토큰 한도 초과 오류 방지)
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
        chunk_size=100
    )
    
    # 벡터 저장소 생성
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
        jj-prog3/streamlit-site
    )

# API 키가 입력되었을 때만 앱 로직 실행
if api_key:
    # --- 캐시 구현 ---
    # 2. API 키가 변경되었는지 확인하고, 변경되었다면 캐시 초기화
    if api_key != st.session_state.cached_api_key:
        st.toast("API 키가 변경되어 답변 캐시를 초기화합니다.")
        st.session_state.query_cache = {}
        st.session_state.cached_api_key = api_key
    # --- 캐시 구현 완료 ---
    
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
                    "date": doc.metadata.get("lastmod", "N_A"), # 'lastmod'가 없을 경우 대비
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
        # 이 함수는 이제 @st.cache_resource를 사용합니다.
        retriever = load_cloudflare_docs(api_key) 
        
        # 5. 사용자 질문 입력
        query = st.text_input("Cloudflare AI 문서에 대해 질문하세요:")

        if query:
            # --- 캐시 구현 ---
            # 3. 캐시에 질문이 있는지 확인
            if query in st.session_state.query_cache:
                st.info("캐시된 답변을 불러왔습니다.")
                result_content = st.session_state.query_cache[query]
                st.markdown(result_content.replace("$", "\$"))
            else:
                # 4. 캐시 미스 (Cache Miss): 체인 실행
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
                    result_content = result.content
                    
                    # 5. 결과를 캐시에 저장
                    st.session_state.query_cache[query] = result_content
                    
                    st.markdown(result_content.replace("$", "\$"))
            # --- 캐시 구현 완료 ---

    except Exception as e:
        st.error(f"문서를 로드하거나 처리하는 중 오류가 발생했습니다: {e}")
        st.warning("API 키가 유효한지, 크레딧이 남아있는지 확인하세요.")


# API 키가 입력되지 않았을 경우 안내 메시지
else:
    st.info("시작하려면 사이드바에 OpenAI API 키를 입력하세요.")