import streamlit as st
from RAG import process_file, setup_chain

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
    retriever = process_file(uploaded_file)
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
