"""
Look into differently structured prompts. 
"""

from langchain.prompts import PromptTemplate

prompt_template = """Use the following pieces of context to extract movies based on the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Helpful Answer:"""


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
