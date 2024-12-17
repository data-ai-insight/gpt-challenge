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
    embeddings = OpenAIEmbeddings()
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
