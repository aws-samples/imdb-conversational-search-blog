"""
Look into differently structured prompts. 
"""

from langchain.prompts import PromptTemplate

prompt_template = """Use the following pieces of context to extract movies based on the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Helpful Answer:"""


instruction_4 = """If I have three columns "title" for movies and "stars" for actors and genres in \\
my opensearch/elasticsearch cluster. 

Can you form a query to search for 50 movies that belong to action and drama and romance genres. 

Output the code snippet in JSON format only and no other additional text"
"""

instruction_3 = """There are seven columns with names given in double quotes "title" for movies and "stars" for actors, "ratings", "directors", "producers","keywords" and "genres" in \\
the opensearch/elasticsearch cluster. 
Here are some examples:
If i want to answer a question "What are the top 25 titles that belong to comedy and action genre". I form the following query
{"query": {"bool": {"must": [{"term": {"genres": "action"}},{"term": {"genres": "comedy"}}]}}, "size": 50,  "sort": [{"rating": {"order": "desc","missing":"_last","unmapped_type" : "long"}}]}

Form a query to answer the following question in quotes

'search Top 75 titles that belong to genre action, romance and comedy  based on rating?' 

Output the code snippet in JSON format only and no other additional text"
"""

instruction_2 = """There are six columns with names given in double quotes "title" for movies, "stars" for actors, "rating", "directors", "producers" and "genres" in \\
the opensearch/elasticsearch cluster. The columns "stars", "director" and "producer" have arrays. Use terms query for columns that have arrays otherwise use term query
Here are some examples:
If i want to answer a question "What are the top 25 titles that belong to comedy and action genre". I form the following query
{"query": {"bool": {"must": [{"term": {"genres": "action"}},{"term": {"genres": "comedy"}}]}}, "size": 50,  "sort": [{"rating": {"order": "desc","missing":"_last","unmapped_type" : "long"}}]}

Another example: If i want to answer a question "Search for all the movies starring Tom Cruise". Use keyword with columns when using terms query.I form the following query with keyword 
{"query": {"bool": {"must": [{"terms": {"<column>.keyword": ["Tom Cruise"]}}]}}}

Sample Terms Query: {
  "query": {
    "terms": {
      "user.id": [ "kimchy", "elkbee" ],
      "boost": 1.0
    }
  }
}

Sample Term Query: {
  "query": {
    "term": {
      "user": "kimchy"
    }
  }
}

For the question given below, do the following
1. Decide what column or columns it maps to
2. Decide whether to use term query, terms query or bool. Use terms query for columns stars, directors, producers. Use <column_name>.keyword
3. Form a query to answer the following question in quotes and output just the query

'Search for movies directed by James Cameron?' 

Show what you have deduced in each step
"""

instruction_1 = """There are six columns with names given in double quotes "title" for movies, "stars" for actors, "rating", "directors", "producers" and "genres" in \\
the opensearch/elasticsearch cluster. The columns "stars", "director" and "producer" have arrays. Use terms query for columns that have arrays otherwise use term query
Here are some examples:
If i want to answer a question "What are the top 25 titles that belong to comedy and action genre". I form the following query
{"query": {"bool": {"must": [{"term": {"genres": "action"}},{"term": {"genres": "comedy"}}]}}, "size": 50,  "sort": [{"rating": {"order": "desc","missing":"_last","unmapped_type" : "long"}}]}

Another example: If i want to answer a question "Search for all the movies starring Tom Cruise". Use keyword with columns when using terms query.I form the following query with keyword 
{"query": {"bool": {"must": [{"terms": {"<column>.keyword": ["Tom Cruise"]}}]}},"sort": [{"rating": {"order": "desc","missing":"_last","unmapped_type" : "long"}}]}

Sample Terms Query: {
  "query": {
    "terms": {
      "user.id": [ "kimchy", "elkbee" ],
      "boost": 1.0
    }
  }
}

Sample Term Query: {
  "query": {
    "term": {
      "user": "kimchy"
    }
  }
}

For the question given below, do the following
1. Decide what column or columns it maps to and whether it requires sorting based on column name
2. For columns stars, directors, producers use `<column_name>.keyword` in the query instead of column_name
3. Decide whether to use term query, terms query or bool. If you use `bool` query, check that you use Terms query for columns "stars", "directors", producers inside the bool.
4. Form a query to answer the following question in quotes

'Search for movies directed by James Cameron and rating greater than 5.5?' 

Show what you have deduced in each step
Answer:
"""

old = """
    Examples are given below with the question on one line and query on the next line
    Question: What are the top 25 movies that belong to comedy and action genre?
    {{"query":{{"bool":{{"must":[{{"terms":{{"genres":["Action"]}}}},{{"terms":{{"genres":["Comedy"]}}}}]}}}}}}

    Question:Search for all the movies starring Tom Cruise and Nicole Kidman with rating greater than 6.0?
    {{"query":{{"bool":{{"must":[{{"terms":{{"stars.keyword":["Tom Cruise"]}},{{stars.keyword:["Nicole Kidman"]}}}},{{"range":{{"rating":{{"gte":6.0}}}}}}]}}}}}}
"""


def get_instruction():
    """
    Instruction for LLM to map user query into domain specific language for Opensearch querying
    Returns:
        str: instruction
    """
    instruction = """There are eight columns with names given in double quotes "title" for movies, "stars" for actors, "rating", "directors", "producers", "keywords", "location" and "genres" in \\
    the opensearch/elasticsearch cluster. The columns "stars", "director", "keywords" and "producer" have arrays so use Terms query otherwise use term query. Use size 15 for all queries unless stated. For columns stars, directors, producers,location use `<column_name>.keyword` in the query instead of column_name.  Use `Terms` query for columns stars, directors, genres, producers and location. If you use `bool` query, check inside the terms query for columns stars, directors, genres, producers and location inside the bool.
   
    Use the following query template to answer the question below. There can be multiple Term or Terms queries inside the "must" array
    {{"query":{{"bool":{{"must":[{{"terms":{{"<column>.keyword":["<entity>"] }}}}]}}}}, "sort": [{{"rating": {{"order": "desc","missing":"_last","unmapped_type" : "long"}}}}]}}
    
    If there are multiple entities in the question use a separate dict in the must clause for each entity. For example, 
    {{"query":{{"bool":{{"must":[{{"terms":{{"<column>.keyword":["<entity1>"]}}}},{{"terms":{{"<column>.keyword":["<entity2>"]}}}}]}}}}, "sort": [{{"rating": {{"order": "desc","missing":"_last","unmapped_type" : "long"}}}}]}}
    
    For example, the query would be {{"query":{{"bool":{{"must":[{{"terms":{{"stars.keyword":["Tom Cruise"]}}}},{{"terms":{{"stars.keyword":["Nicole Kidman"]}}}}]}}}}}}

    For the question given below, use the query do the following
    1. Extract the entities from the question and map each entity to columns "stars","directors", "producers, "keywords" and "genres"
    2. Form a query to answer the following question in double quotes based on the query template above and no other information. "must" clause inside the query with arrays must have a single entity only.

    "{0}"

    Show what you have deduced in each step
    Answer:
    """
    return instruction


def get_chat_prompt():
    """
    Prompt for the chat capability
    Returns:
        langchain.prompts.PromptTemplate: prompt template for chat
    """
    template = """You are a chatbot having a conversation with a human.
    Given the following extracted parts of a long document and a question, create a final answer.

    {context}

    {chat_history}
    (Use only the above information to answer the question. )
    Human: {human_input}
    Chatbot:"""

    CHAT_PROMPT = PromptTemplate(
        input_variables=["chat_history", "human_input", "context"], template=template
    )

    return CHAT_PROMPT


QA_PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
    validate_template=False,
)

CHAT_PROMPT = get_chat_prompt()
