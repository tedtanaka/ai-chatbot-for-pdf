"""Search a query in vector db, get relevant text chunks,
feed it to th LLM and generate a concise response"""
import os

from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Milvus
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def get_db_connection(collection_name):
    """Returns a Milvus DB connection object"""
    embeddings = OpenAIEmbeddings()
    return Milvus(embeddings, connection_args={
        "user": os.getenv("MILVUS_DB_USERNAME"),
        "password": os.getenv("MILVUS_DB_PASSWORD"),
        "host": os.getenv("MILVUS_DB_HOST"),
        "port": os.getenv("MILVUS_DB_PORT"),
        "db_name": os.getenv("MILVUS_DB_NAME")},
                  collection_name=os.getenv("MILVUS_DB_COLLECTION"))

def get_similar_docs(query: str):
    """Fetches similar text from the vector db"""
    vector_db = get_db_connection("my_collection")
    return vector_db.similarity_search_with_score(query, k=3)


def fetch_answer_from_llm(query: str):
    """Fetches relevant answer from LLM"""
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                     temperature=0.6,
                     max_tokens=1024)
    chain = load_qa_chain(llm, "stuff")
    similar_docs = get_similar_docs(query)
    docs = []
    for doc in similar_docs:
        docs.append(doc[0])
    chain_response = chain.invoke(input={"input_documents": docs, "question": query})
    return chain_response["output_text"]


def generate_answer():
    """Gets an answer to the user query"""
    try:
        query = input("Enter your query: ")
        answer = fetch_answer_from_llm(query)
        print(answer)
        return
    except Exception as exception_message:
        print(str(exception_message))


if __name__ == "__main__":
    generate_answer()