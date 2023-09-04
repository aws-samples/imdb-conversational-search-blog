import yaml

import platform

from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.memory import ConversationBufferMemory

import sys

sys.path.append("../src/")
from src.search import submit_results
from src.endpoint import amazon_bedrock_embeddings
from src.movie_utils import describe_movie, get_trending_content

with open("../config.yml", "r") as file:
    config = yaml.safe_load(file)


def validate_environment():
    assert platform.python_version() >= "3.10.6"


def amazon_opensearch_docsearch(aos_config, docs, embeddings):
    """
    Perform document search from opensearch
    Args:
        aos_config(dict): configurations for host and index
        docs(list): list of docs to search
        embeddings(list): list of embeddings
    Return
        langchain.vectorstores.opensearch_vector_search.OpenSearchVectorSearch:
            opensearch vector search output
    """
    _aos_host = aos_config["aos_host"]
    _aos_index = aos_config["aos_index"]

    docsearch = OpenSearchVectorSearch.from_texts(
        texts=[d.page_content for d in docs],
        embedding=embeddings,
        metadatas=[d.metadata for d in docs],
        opensearch_url=[{"host": _aos_host, "port": 443}],
        index_name=_aos_index,
        http_auth=(aos_config["aos_user"], aos_config["aos_pass"]),
        use_ssl=True,
        pre_delete_index=True,
    )
    return docsearch


def create_vector_store(docs, aos_config):
    """
    Create open search vector store
    Args:
        aos_config(dict): configurations for host and index
        docs(list): list of docs to search
    Return
        langchain.vectorstores.opensearch_vector_search.OpenSearchVectorSearch:
            opensearch vector search output
    """
    embeddings = amazon_bedrock_embeddings()

    return amazon_opensearch_docsearch(
        aos_config=aos_config, docs=docs, embeddings=embeddings
    )


def chain_qa(llm, prompt=None, verbose=False):
    """
    Get QnA chain for given QnA prompt
    Args:
        llm(Langchain): large language model type
        prompt(Langchain.prompt): input question and context
        verbose(boolean): whether to print out everything or not
    Return:
        langchain.chains.question_answering: QA chain for llm
    """
    return load_qa_chain(llm, chain_type="stuff", verbose=verbose, prompt=prompt)


def chain_chat(llm, prompt=None, verbose=False):
    """
    Get conversational chain for given chat prompt
    Args:
        llm(Langchain): large language model type
        prompt(Langchain.prompt): input question and context
        verbose(boolean): whether to print out everything or not
    Return:
        langchain.chains.question_answering: QA chain for llm
    """
    memory = ConversationBufferMemory(
        memory_key="chat_history", input_key="human_input"
    )
    return load_qa_chain(
        llm, chain_type="stuff", memory=memory, verbose=verbose, prompt=prompt
    )


def get_docs(response, topk=5):
    """
    Create langchain docs for movie description
    Args:
        response(dict): information about all movies (actors, title, etc..)
        topk(int): Number of movies to include docs
    Returns:
        list(str): list of aggregated information about each movie
    """
    docs = [
        Document(
            page_content=describe_movie(resp["_source"]),
            metadata={"source": resp["_source"]["title"]},
        )
        for resp in response[0:topk]
    ]
    return docs


def search_and_answer(store, llm, query, ops, embedding_model, task=None, k=10):
    """
    Process user query for search and chat.
    Args:
        store(OpenSearchVectorStore): vector store
        llm(Langchain LLM): llm to perform search
        query(str): question asked by the user
        ops(OpenSearch): initialized opensearch instance
        embedding_model(SentenceTransformer/BedrockEmbeddings): llm embedding model
        k(int): number of movies to output from similarity search
        task(str): whether it is a Search or Search and Chat task as given by the user
    Returns:
        dict: outputted movies from similarity search and corresponding information
    """

    if isinstance(store, OpenSearchVectorSearch):
        docs = store.similarity_search(
            query,
            k=k,
            # include_metadata=False,
            verbose=False,
        )

    if "Search" in task:
        if "trend" in query:
            response = get_trending_content(config["TMDB_API_TOKEN"])
            short_response = [
                (
                    h["_source"]["title"],
                    h["_source"]["poster_url"],
                    h["_id"],
                    h["trailer"],
                )
                for h in response
            ]
        else:
            response = submit_results(llm, query, ops, embedding_model)
            response = [hit for hit in response["hits"]["hits"]]
            short_response = [
                (h["_source"]["title"], h["_source"]["poster_url"], h["_id"], None)
                for h in response
            ]

        if "Chat" in task:
            docs = get_docs(response, 5)
            return {"response": short_response, "docs": docs}

        else:
            return {"response": short_response, "docs": None}

    else:
        assert "Task Not Found"
