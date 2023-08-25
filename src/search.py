import numpy as np
import json

import boto3
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection

import ai21
from src.prompt import get_instruction


# Comprehend constant
REGION = "us-east-1"


def initialize_ops(
    host="vpc-llm-imdb-search-wdv7w5ydpuuzhekveyxmnhsnw4.us-east-1.es.amazonaws.com",
    port=443,
    region="us-east-1",
    service="es",
):
    """
    Initialize opensearch instance with specific host, port, and region
    Args:
        host(str): opensearch host
        port(int): port number
        region(str): aws region
        service(str): service type
    Returns:
        OpenSearch: initialized opensearch instance
    """
    # For example, my-test-domain.us-east-1.es.amazonaws.com  # e.g. us-west-1
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        service,
        session_token=credentials.token,
    )

    ops = OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=10,
    )

    return ops


def get_results(ops, index_name, search_text, num_results):
    """
    Format query and use opensearch to search the query
    Args:
        ops(OpenSearch): initialized opensearch instance
        index_name(str): opensearch instance name
        search_text(str): search query
        num_results(int): number of movies to output
    Returns:
        list(dict): results from opensearch search using the query from search text
    """
    query = {
        "size": num_results,
        "query": {
            "multi_match": {
                "query": f'"{search_text}"',
                "fields": [
                    "stars",
                    "title",
                    "location",
                    "genres",
                    "directors",
                    "producers",
                ],
            }
        },
    }
    print(f"Search Text: {search_text}, Index: {index_name} and Query: {query}")

    results = ops.search(index=index_name, body=query)
    return results


def query_matches(tokens, doc):
    """
    Find matches of token in a document key.
    Args:
        tokens(list): list of tokens to find counts in documents
        doc(list): list of documents
    Return:
        Integer: counts of each token in documents
    """
    count = 0
    string_values = set(doc.keys()).difference(
        set(["rating", "embeddings", "year", "plotLong"])
    )

    for key in string_values:
        if isinstance(doc[key], str):
            doc[key] = doc[key].lower()
        else:
            doc[key] = [value.lower() for value in doc[key]]

        for token in tokens:
            if token in doc[key]:
                count += 1

    return count


def filter_results(search_text, results, topk):
    """
    Filter opensearch results based on the search text
    Args:
        search_text(str): text query
        results(dict): opensearch results
        topk(int): number of movies to output
    Returns:
        list: information about outputted movies
    """
    matches = []
    tokens = get_tokens(search_text.lower())
    for result in results["hits"]["hits"]:
        match = query_matches(tokens, result["_source"])
        matches.append(match)

    final_idx = np.argsort(-np.array(matches))[:topk]
    docs = [results["hits"]["hits"][idx]["_source"] for idx in final_idx]
    return docs


def process_docs(docs):
    """
    Get title, genre, stars, etc.. for each movie
    Args:
        docs(list): list of information for each movie
    Returns:
        list: get necessary information about each movie
    """
    keys = [
        "title",
        "genres",
        "stars",
        "directors",
        "producers",
        "location",
        "plotLong",
    ]
    short_doc = []
    for doc in docs:
        new_doc = {}
        for key in keys:
            new_doc[key] = doc.get(key, "")
        short_doc.append(new_doc)

    return short_doc


def detect_dominant_language(text):
    """
    Function for detecting the dominant language from an input text
    Args:
        text(str): input text
    Returns:
        dict: response type from comprehend
    """
    comprehend = boto3.client("comprehend", region_name=REGION)
    response = comprehend.detect_dominant_language(Text=text)
    return response


def detect_entities(text, language_code):
    """
    Function for detecting named entities from an input text
    Args:
        text(str): input text
        language_code(str)
    Returns:
        dict: detect all entities using comprehend
    """
    comprehend = boto3.client("comprehend", region_name=REGION)
    response = comprehend.detect_entities(Text=text, LanguageCode=language_code)
    return response


def detect_key_phraes(text, language_code):
    """
    Function for detecting key phrases from an input text
    Args:
        text(str): input text
        language_code(str)
    Returns:
        dict: detect key phrases using comprehend
    """
    comprehend = boto3.client("comprehend", region_name=REGION)
    response = comprehend.detect_key_phrases(Text=text, LanguageCode=language_code)
    return response


# Function for detecting sentiment
def detect_sentiment(text, language_code):
    """
    Function for detecting key phrases from an input text
    Args:
        text(str): input text
        language_code(str)
    Returns:
        dict: detect sentiment using comprehend
    """
    comprehend = boto3.client("comprehend", region_name=REGION)
    response = comprehend.detect_sentiment(Text=text, LanguageCode=language_code)
    return response


def get_tokens(text):
    """
    From text, retrieve named entities and key phrases
    Args:
        text(str): text to get entities and key phrases from
    Returns:
        list(set): all tokens
    """
    # language code
    language_code = "en"
    tokens = []
    # detecting named entities
    result = detect_entities(text, language_code)
    tokens.extend([entity["Text"] for entity in result["Entities"]])
    # detecting key phrases
    result = detect_key_phraes(text, language_code)
    tokens.extend([entity["Text"] for entity in result["KeyPhrases"]])
    return list(set(tokens))


def get_docs(ops, index_name, search_text, num_results, topk):
    """
    Retrieve docs for a given search text
    Args:
        ops(OpenSearch): initialized opensearch index
        index_name(str): name of the opensearch index
        search_text(str): text_query
        num_results(int)
        topk(int): how many results to output
    Return:
        list: list of retrieved documents
    """
    results = get_results(ops, index_name, search_text, num_results)
    docs = filter_results(search_text, results, topk)
    docs = process_docs(docs)

    return docs


def extract_json(x):
    """
    Extract json object within input
    Args:
        x(str): input json object
    Return:
        str: components within the json
    """
    for i, j in enumerate(x):
        if j == "{":
            break
    for j, k in enumerate(list(reversed(x))):
        if k == "}":
            break
    return x[i : len(x) - j]

def get_exact_match(llm, question):
    """
    Get result from opensearch based on the input question
    Args:
        llm(Jurassic Jumbo Instruct/Bedrock): user suggested llm
        question(str): user inputted question
    Returns:
        list(dict): opensearch results
    """
    instruction = get_instruction()

    instruction = instruction.format(question)
    query = ask_questions_using_prompt(llm, instruction).replace("\n", "").strip()
    query = extract_json("".join(query.split("2.")[-1]))
    query = json.loads(query)

    if query.get("size", "Not Found") == "Not Found":
        query["size"] = 50

    if query.get("sort", "Not Found") == "Not Found":
        query["sort"] = [
            {"rating": {"order": "desc", "missing": "_last", "unmapped_type": "long"}}
        ]

    ops = initialize_ops()
    results = ops.search(index="imdb_small_posters", body=query)
    return results

def search_movie(query, ops):
    """
    Get the movie names from opensearch results
    Args:
        query(str): user inputted query
        ops(Opensearch): initialized opensearch instance
    Returns:
        list: list of movie names
    """
    results = ops.search(index="imdb_small_posters", body={"query":{"bool":{"must":[{"terms":{"title.keyword":[query]}}]}}})
    return results['hits']['hits'][0]['_id'] if results else None

def get_movie_emb(ttid):
    """
    Get movie embeddings
    Args:
        ttid(str): movie id
    Returns:
        list: movie embedding
    """
    movie_emb = movies_emb_dict[ttid]
    return movie_emb


def get_similar_movies(query, ops, topk=10):
    """
    Obtain similar movies query
    Args:
        query(str): user inputted query
        ops(Opensearch): opensearch initialized instance
        topk(integer): number of movies to output
    Returns:
        list: top responses from opensearch
    """
    movie_id = search_movie(query, ops)
    if movie_id: 
        movie_emb = get_movie_emb(movie_id)
        response = get_relevant_plots(movie_emb, ops)
        response = [hit["_source"] for hit in response["hits"]["hits"] if hit["_source"]['titleId'] != movie_id]
        top_response = sorted(response, key=lambda d: float(d['rating']), reverse=True) 
        top_response = [{'_source': resp, '_id': resp['titleId']} for resp in top_response]
        return top_response[:topk]

    else:
        return None

def submit_results(llm, query, ops, embedding_model):
    """
    Get semantic match of the query from opensearch
    Args:
        llm(Langchain LLM): llm to perform search
        query(str): input question
        ops(OpenSearch): initialized opensearch index
        embedding_model(Langchain embed): embedding model (sentence transformer)
    Returns:
        list(dict): opensearch output of movies based on user qyery
    """
    exact_match_candidates = ['shot', 'location', 'actor', 'star', 'plot', 'story', 'rating', 'direct', 'produce']
    similar_candidates = ['movies similar', 'movies like']
    exact_match, similarity_match = False, False
    for word in similar_candidates: 
        if word in query: similarity_match=True
    for word in exact_match_candidates: 
        if word in query: exact_match=True
    
    if similarity_match:
        response = get_similar_movies(query.split('to')[-1].strip(), ops)
    
    elif exact_match:
        response = get_exact_match(llm, query)
    else:
        response = get_semantic_match(query, ops, embedding_model)
    
    return response
    

def ask_questions_using_prompt(llm, prompt_template):
    """
    Get the llm output of the prompt template
    Args:
        llm(Jurassic Jumbo Instruct/Bedrock): user suggested llm
        prompt_template(langchain.prompts.promptTemplate): prompt template to convert user
            query to opensearch domain specific language query
    Return:
        str: llm outputted Opensearch DSL query
    """
    text = llm(prompt_template)
    return text

def get_embedding(model, text):
    """
    Embed input text by given model
    Args:
        model(SentenceTransformer/BedrockEmbeddings): embedding model
        text(str): text to embed
    Returns:
        list: array of embeddings for the query
    """
    return model.embed_query(text)


def get_relevant_plots(query_emb, ops, topk=5):
    """
    Retrieve all relevant information
    Args:
        query_emb(list): embedding of the query
        ops(Opensearch): opensearch index
    Return:
        list(dict): Opensearch search result
    """
    columns = [
        "directors",
        "producers",
        "stars",
        "rating",
        "location",
        "genres",
        "year",
        "plotLong",
        "plot",
        "title",
        "poster_url",
        "titleId",
        "keyword",
    ]
    query = {"size": topk, "query": {"knn": {"emb": {"vector": query_emb, "k": topk}}}}

    res = ops.search(index="imdb_knn", body=query, stored_fields=columns)

    for resp in res["hits"]["hits"]:
        resp["_source"] = resp.pop("fields")
        for col in ["year", "titleId", "rating", "plotLong", "poster_url"]:
            resp["_source"][col] = resp["_source"][col][0]

    return res


def get_semantic_match(query, ops, embedding_model, topk=10):
    """
    Get matches of the query from opensearch
    Args:
        query(str): input question
        ops(Opensearch): initialized opensearch index
        embedding_model(Langchain.embedding):
        topk(int): how many results to output
    Returns:
        list(dict): Opensearch search result of query
    """
    emb = get_embedding(embedding_model, query)
    plot_response = get_relevant_plots(emb, ops, topk=topk)

    return plot_response
