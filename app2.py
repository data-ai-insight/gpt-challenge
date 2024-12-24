import streamlit as st
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.utilities.wikipedia import WikipediaAPIWrapper

# Streamlit 설정
st.set_page_config(page_title="QuizGPT+", page_icon="❓", layout="wide")

st.title("QuizGPT+")
st.markdown("""
이 앱은 파일 업로드 또는 키워드 검색을 통해 사용자 지정 퀴즈를 생성합니다.  
문서를 업로드하거나 키워드를 입력해보세요!
""")

# Sidebar 입력
with st.sidebar:
    st.header("설정")
    api_key = st.text_input("OpenAI API 키 입력", type="password")
    difficulty = st.selectbox("시험 난이도 선택", ["쉬움", "보통", "어려움"])
    st.markdown("[코드 확인하기](https://github.com/username/repo)")

    # 문서 업로드 또는 키워드 검색
    uploaded_file = st.file_uploader("문서를 업로드하세요", type=["txt", "pdf", "docx"])
    search_keyword = st.text_input("키워드 입력 (문서 업로드 없이 사용 가능)")

if not api_key:
    st.error("OpenAI API 키를 입력하세요.")
    st.stop()

# LLM 및 기능 정의
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=api_key,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

questions_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", f"""
        당신은 교사로 역할하는 AI 비서입니다.
        제공된 컨텍스트를 바탕으로 난이도 '{difficulty}'에 맞게 10개의 퀴즈를 생성하세요.
        각 문제는 하나의 정답과 세 개의 오답을 포함해야 합니다.
        답안 형식은 다음과 같습니다:
        
        Question: {{"문제 내용"}}
        Answers: 오답1|오답2|정답(o)|오답3
        
        컨텍스트: {{context}}
        """),
    ]
)

# 파일 처리
def process_file(file):
    loader = UnstructuredFileLoader(file.name)
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

# 퀴즈 생성 및 출력
def generate_quiz(context):
    response = questions_prompt.invoke(context=context, llm=llm)
    return response

# 위키피디아 검색
def search_wikipedia(keyword):
    wiki = WikipediaAPIWrapper()
    result = wiki.run(keyword)
    return result

context = ""

if uploaded_file:
    docs = process_file(uploaded_file)
    context = "\n\n".join([doc.page_content for doc in docs])
    st.success("문서에서 컨텍스트를 추출했습니다.")

elif search_keyword:
    context = search_wikipedia(search_keyword)
    st.success(f"키워드 '{search_keyword}'에 대한 정보를 검색했습니다.")

if context:
    st.info("퀴즈를 생성하는 중입니다. 잠시만 기다려주세요...")
    quiz = generate_quiz(context)

    # 퀴즈 출력
    if quiz:
        st.success("퀴즈 생성 완료!")
        st.markdown("### 퀴즈")
        for question in quiz["questions"]:
            st.write(f"**{question['question']}**")
            for answer in question["answers"]:
                st.radio("정답을 선택하세요.", [answer for answer in question["answers"]])
        if st.button("결과 제출"):
            st.balloons()
            st.success("축하합니다! 만점입니다!")
else:
    st.info("문서를 업로드하거나 키워드를 입력하세요.")
