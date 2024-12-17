import streamlit as st
import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def process_file(file):
    """
    파일을 로드하고 FAISS 벡터 스토어를 생성합니다.
    """
    # 파일 저장
    file_content = file.read()
    file_path = f"./uploaded_files/{file.name}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 문서 로드 및 분할
    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(separator="\n", chunk_size=600, chunk_overlap=100)
    docs = loader.load_and_split(text_splitter=splitter)

    # 임베딩 및 캐싱
    cache_dir = LocalFileStore(f"./.cache/{file.name}")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    # 벡터 스토어 생성
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def setup_chain(retriever, api_key):
    """
    Conversational Retrieval Chain 설정.
    """
    # OpenAI LLM 설정
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)

    # 메모리 설정
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True, 
        output_key="answer"
    )

    # 체인 생성
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    return chain

# Streamlit 페이지 설정
st.set_page_config(
    page_title="RAG Pipeline",
    page_icon="📄",
    layout="wide"
)

# 제목 및 설명
st.title("RAG Pipeline with Streamlit 📄")
st.markdown("""
이 앱은 문서를 분석하고 AI를 통해 질문에 답변을 생성하는 RAG (Retrieval-Augmented Generation) 파이프라인입니다.  
문서를 업로드하고 질문을 입력해보세요!
""")

# Sidebar 설정
with st.sidebar:
    st.header("설정")
    # OpenAI API 키 입력
    api_key = st.text_input("OpenAI API Key", type="password")
    st.write("---")
    # 파일 업로드
    uploaded_file = st.file_uploader("문서를 업로드하세요 (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
    # GitHub 링크
    github_link = "https://github.com/data-ai-insight/gpt-challenge.git"
    st.markdown(f"[🔗 GitHub 리포지토리]({github_link})")

# API 키 입력 확인
if not api_key:
    st.warning("OpenAI API 키를 입력하세요.")
    st.stop()

# 파일 업로드 확인 및 처리
if uploaded_file:
    st.info("파일을 처리 중입니다. 잠시만 기다려주세요...")
    retriever = process_file(uploaded_file, api_key)
    chain = setup_chain(retriever, api_key)
    st.success("파일 처리가 완료되었습니다! 질문을 시작하세요.")

    # 채팅 세션 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 채팅 기록 표시
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    user_input = st.chat_input("파일에 대해 궁금한 점을 질문하세요...")
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
    st.info("문서를 업로드하면 AI와 대화할 수 있습니다.")
