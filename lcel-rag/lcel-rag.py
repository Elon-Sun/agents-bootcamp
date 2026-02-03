from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import Client
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model="gpt-4o-mini")
no_rag_chain = llm | StrOutputParser()
print("=== No RAG ===")
result1 = no_rag_chain.invoke("What is OpenClaw?")
print(result1)


loader = UnstructuredMarkdownLoader("./README.md")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20, add_start_index=True)
all_splits = text_splitter.split_documents(docs)

vector_store = Chroma.from_documents(
    documents=all_splits,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_embeddings"
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})

client = Client()
prompt = client.pull_prompt("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\n=== RAG ===")
print("Prompt: " + prompt.messages[0].prompt.template + "\n")
questions = ["Explain OpenClaw in detail.", "How to install OpenClaw?", "How does OpenClaw work?"]
for question in questions:
    print(f"Question: {question}")
    result2 = rag_chain.invoke(question)
    print(result2 + "\n")

another_prompt = client.pull_prompt("ahmedghani/agentic-rag")
another_rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | another_prompt
    | llm
    | StrOutputParser()
)
print("\n=== Another RAG Prompt ===")
print("Prompt: " + another_prompt.messages[0].prompt.template + "\n")
for question in questions:
    print(f"Question: {question}")
    result3 = another_rag_chain.invoke(question)
    print(result3+ "\n")