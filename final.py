import streamlit as st
import openai
import time
import json
from typing import Optional, List, Dict, Any
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# 페이지 설정
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔍",
    layout="wide"
)

# 도구 함수들
def search_web(inputs: Dict[str, Any]) -> str:
    """DuckDuckGo로 웹 검색을 수행합니다."""
    try:
        ddg = DuckDuckGoSearchAPIWrapper()
        return ddg.run(inputs["query"])
    except Exception as e:
        return f"Search failed: {str(e)}"

# 도구 함수 매핑
tool_functions = {
    "search_web": search_web,
}

# Assistant 도구 정의
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web using DuckDuckGo search engine",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Assistant 생성 함수
def create_assistant(client: openai.Client) -> Any:
    """OpenAI Assistant 생성"""
    try:
        assistant = client.beta.assistants.create(
            name="Research Assistant",
            instructions="""You are a helpful research assistant that helps users find and analyze information. 
                When researching a topic:
                1. Search multiple sources using the provided tools
                2. Always cite your sources with URLs
                3. Structure your response with clear sections
                4. Provide a balanced view of the topic
                5. Include any relevant technical details
                6. End with actionable recommendations when appropriate""",
            model="gpt-3.5-turbo",
            tools=tools
        )
        return assistant
    except Exception as e:
        st.error(f"Failed to create assistant: {str(e)}")
        return None

# Thread 관리 함수들
def create_thread(client: openai.Client) -> Optional[Any]:
    """새로운 Thread 생성"""
    try:
        return client.beta.threads.create()
    except Exception as e:
        st.error(f"Failed to create thread: {str(e)}")
        return None

def process_tool_calls(client: openai.Client, run: Any, thread_id: str) -> List[Dict[str, str]]:
    """도구 실행 결과 처리"""
    try:
        outputs = []
        for tool_call in run.required_action.submit_tool_outputs.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            if function_name in tool_functions:
                result = tool_functions[function_name](arguments)
                outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": str(result)  # 결과를 문자열로 변환
                })
        return outputs
    except Exception as e:
        st.error(f"Tool execution failed: {str(e)}")
        return []

def wait_for_run_completion(client: openai.Client, thread_id: str, run_id: str, timeout: int = 300) -> Optional[Any]:
    """Run 완료 대기"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            run = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
            
            if run.status == "completed":
                return run
            elif run.status == "requires_action":
                # 도구 실행이 필요한 경우
                outputs = process_tool_calls(client, run, thread_id)
                if outputs:
                    client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread_id,
                        run_id=run_id,
                        tool_outputs=outputs
                    )
            elif run.status in ["failed", "cancelled", "expired"]:
                st.error(f"Run failed with status: {run.status}")
                return None
                
            time.sleep(1)
        except Exception as e:
            st.error(f"Error while waiting for run completion: {str(e)}")
            return None
            
    st.error("Request timed out")
    return None

def process_message(
    client: openai.Client,
    thread_id: str,
    assistant_id: str,
    user_message: str
) -> Optional[str]:
    """메시지 처리 및 응답 생성"""
    try:
        # 사용자 메시지 추가
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_message
        )
        
        # Run 생성
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )
        
        # Run 완료 대기 및 처리
        completed_run = wait_for_run_completion(client, thread_id, run.id)
        if not completed_run:
            return None
            
        # 응답 메시지 가져오기
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        if messages.data:
            return messages.data[0].content[0].text.value
        return None
        
    except Exception as e:
        st.error(f"Failed to process message: {str(e)}")
        return None

# Streamlit UI
st.title("Research Assistant")
st.markdown("""
This AI assistant helps you research topics by searching the web and providing detailed, 
well-organized information. Enter your question below to get started.
""")

# Sidebar 설정
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    
    st.markdown("---")
    st.markdown("### Project Links")
    st.markdown("[GitHub Repository](https://github.com/yourusername/research-assistant)")
    
# Session state 초기화
if "assistant" not in st.session_state:
    st.session_state.assistant = None
if "thread" not in st.session_state:
    st.session_state.thread = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# API 키 확인
if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    st.stop()

try:
    # OpenAI 클라이언트 초기화
    client = openai.Client(api_key=api_key)
    
    # Assistant와 Thread 초기화
    if st.session_state.assistant is None:
        with st.spinner("Initializing assistant..."):
            st.session_state.assistant = create_assistant(client)
    if st.session_state.thread is None:
        with st.spinner("Creating new thread..."):
            st.session_state.thread = create_thread(client)
            
    if st.session_state.assistant and st.session_state.thread:
        # 채팅 기록 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 사용자 입력 처리
        user_input = st.chat_input("Ask your research question...")
        if user_input:
            # 사용자 메시지 표시
            st.chat_message("user").markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # 응답 생성
            with st.spinner("Researching and generating response..."):
                response = process_message(
                    client,
                    st.session_state.thread.id,
                    st.session_state.assistant.id,
                    user_input
                )
                
                if response:
                    # AI 답변 표시
                    st.chat_message("assistant").markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error("Failed to generate response. Please try again.")
    else:
        st.error("Failed to initialize the assistant or thread. Please check your API key and try again.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")