import streamlit as st
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# 페이지 설정
st.set_page_config(
    page_title="Cloudflare SiteGPT",
    page_icon="☁️",
    layout="wide"
)

# 제목 및 설명
st.title("Cloudflare SiteGPT")
st.markdown("""
Cloudflare 공식 문서에서 질문에 답변하는 챗봇입니다.

지원되는 제품:
- [AI Gateway](https://developers.cloudflare.com/ai-gateway/)
- [Vectorize](https://developers.cloudflare.com/vectorize/)
- [Workers AI](https://developers.cloudflare.com/workers-ai/)
""")

# Sidebar 설정
with st.sidebar:
    st.header("설정")
    # OpenAI API 키 입력
    api_key = st.text_input("OpenAI API Key", type="password")
    st.write("---")
    # 사이트맵 URL 입력
    sitemap_url = st.text_input(
        "Cloudflare Sitemap URL", 
        placeholder="https://developers.cloudflare.com/sitemap-0.xml"
    )
    # 필터링할 키워드 입력 (예: 특정 문서만 가져오도록)
    filter_keyword = st.text_input("문서 필터링 키워드", placeholder="예: ai-gateway")

# XML 사이트맵에서 문서를 로드하는 함수
@st.cache_data(show_spinner="웹사이트 데이터를 로드 중입니다...")
def load_docs_from_sitemap(sitemap_url, api_key, filter_keyword=None):
    """
    사이트맵에서 문서를 로드하고, 필터링하는 기능을 추가합니다.
    캐시 키에 sitemap_url과 filter_keyword를 포함시켜, 변경되지 않으면 캐시를 사용하도록 합니다.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(sitemap_url)
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)

    # 필터링할 키워드가 있을 경우 필터링
    if filter_keyword:
        docs = [doc for doc in docs if filter_keyword.lower() in doc.page_content.lower()]

    # 벡터 스토어 생성
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store.as_retriever()

# Conversational Retrieval Chain 설정 함수
def setup_chain(retriever, api_key):
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0, openai_api_key=api_key
    )
    
    # 메모리 설정
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True, 
        output_key="answer"  # output_key를 명시적으로 설정
    )
    # 체인 생성
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    return chain

# API 키 입력 확인
if not api_key:
    st.warning("OpenAI API 키를 입력하세요.")
    st.stop()

# 사이트맵 URL 확인 및 처리
if sitemap_url:
    if ".xml" not in sitemap_url:
        st.error("유효한 Sitemap URL을 입력하세요.")
    else:
        with st.spinner("사이트맵에서 데이터를 로드 중입니다..."):
            retriever = load_docs_from_sitemap(sitemap_url, api_key, filter_keyword)
            chain = setup_chain(retriever, api_key)
            st.success("문서 로드가 완료되었습니다! 질문을 입력하세요.")

        # 채팅 세션 초기화
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # 채팅 기록 표시
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 사용자 입력 처리
        user_input = st.chat_input("Cloudflare 문서에 대해 궁금한 점을 입력하세요...")
        if user_input:
            # 사용자 메시지 표시
            st.chat_message("user").markdown(user_input)
            st.session_state["messages"].append({"role": "user", "content": user_input})

            # AI 답변 생성
            with st.spinner("답변을 생성 중입니다..."):
                response = chain.invoke({"question": user_input})
                answer = response["answer"]

            # AI 답변 표시
            st.chat_message("assistant").markdown(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer})
else:
    st.info("사이트맵 URL을 입력하면 문서 기반 AI와 대화할 수 있습니다.")
