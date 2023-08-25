"""
Version 2: UI format of the IMDB demo including choices given to the user for the LLM to use, that task to perform, questions to ask, and many more. 
"""
import requests
import yaml

import sys
import base64

from prompt import QA_PROMPT, CHAT_PROMPT
from search import launch_encoder, initialize_ops
import langchain_qa_chat

import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header

sys.path.append("../src/")
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

# task
_TASK_RE = False
st.set_page_config(layout="wide")


@st.cache_resource
def check_env():
    langchain_qa_chat.validate_environment()


@st.cache_data
def list_llm_models():
    return [
        "Jurassic-Jumbo-Instruct",
        "Text2Text",
        "FLAN-T5-XXL",
    ]
    # return ['Bedrock','FLAN-T5-XXL']


@st.cache_data
def list_columns():
    return ["stars", "directors", "producers", "genre"]


def list_task_type():
    return ["Search", "Search & Chat!"]


@st.cache_data
def get_as_base64(url):
    return base64.b64encode(requests.get(url).content).decode()


@st.cache_data
def get_img_with_href(img_url, target_url):
    html_code = f"""
        <a href="{target_url}">
            <img src="{img_url}" height="200"/>
        </a>"""
    return html_code


# @st.cache_data
def recommended_k():
    return 10


@st.cache_data
def list_questions():
    return [
        "What are the movies starring Tom Cruise?",
        "what are some sniper action movies?",
        "What are the movies directed by James Cameron and rating greater than 5.5?",
        "What are the movies starring Kate Winslet and starring Leonardo DiCaprio?",
        "Ask your question",
    ]


@st.cache_resource
def create_qa_chain(model, os_task_type, prompt=None, verbose=False):
    if model == "FLAN-T5-XXL":
        llm = langchain_qa_chat.sagemaker_endpoint(config["llm"]["t5_endpoint"])
    elif model == "Bedrock":
        llm = langchain_qa_chat.amazon_bedrock_llm(verbose=verbose)
    elif model == "Text2Text":
        llm = langchain_qa_chat.text2text_llm()
    elif model == "Jurassic-Jumbo-Instruct":
        llm = langchain_qa_chat.sagemaker_endpoint_ai21(config["llm"]["ai21_instruct"])
    else:
        assert False
    print("Make QA chain for", model)
    search_chain = langchain_qa_chat.chain_qa(llm, verbose=verbose, prompt=QA_PROMPT)
    chat_chain = None
    if "Chat" in os_task_type:
        chat_chain = langchain_qa_chat.chain_chat(
            llm, verbose=verbose, prompt=CHAT_PROMPT
        )

    ops = initialize_ops()
    embedding_model = launch_encoder()

    return search_chain, chat_chain, ops, embedding_model


def get_text():
    """
    Function for taking user provided prompt as input
    """
    input_text = st.text_input(
        "Chat with AI to refine your choices", "", key="text_input"
    )
    return input_text


def main():
    check_env()

    # col_title = st.columns(3)
    # col_title[1].title("IMDb Conversational Search")
    st.header("IMDb Conversational Search")

    # col_model = st.columns(3)
    # model = col_model[1].selectbox("Select LLM", list_llm_models())
    model = "Jurassic-Jumbo-Instruct"

    # os_task_type = st.columns(3)[1].selectbox("Task Type", list_task_type())
    os_task_type = "Search and Chat!"

    col_query = st.columns(3)
    query = col_query[0].selectbox("Select question", [""] + list_questions())
    if "Ask" in query:
        query = st.columns(3)[0].text_input(
            "Your Question: ", placeholder="Ask me anything ...", key="input"
        )
    if not query:
        return

    k = recommended_k()
    answer = "Not found"
    print("Q:", query)
    store = "CUSTOM"

    for attempt in range(1):
        try:
            search_chain, chat_chain, ops, embedding_model = create_qa_chain(
                model, os_task_type, verbose=True
            )

            response = langchain_qa_chat.search_and_answer(
                store,
                search_chain,
                query,
                ops,
                embedding_model,
                k=k,
                task=os_task_type,
                doc_source_contains=None,
            )
            print(answer)
            answer = response["response"]
            break  # Success
        except Exception as e:
            print(e)
            st.spinner(text=type(e).__name__)
            if type(e).__name__ == "ValidationException" and k > 1:
                print("Retrying using shorter context")
                k -= 1
            elif type(e).__name__ == "ThrottlingException":
                print("Retrying")
            else:
                # continue
                raise e

    if "Search" in os_task_type:
        if len(answer) > 0:
            cols = st.columns(10)
            for i, (title, poster, ttid) in enumerate(answer[0:10]):
                # print(ttid)

                imdb_url = f"https://www.imdb.com/title/{ttid}/"
                # print(imdb_url)
                try:
                    image_html = get_img_with_href(poster, imdb_url)
                    # print(image_html)
                    cols[i % 10].markdown(image_html, unsafe_allow_html=True)
                except requests.exceptions.MissingSchema:
                    cols[i % 10].markdown("")
                cols[i % 10].markdown(f"[{title}]({imdb_url})")
                if i % 10 == 0 and i > 1:
                    cols = st.columns(10)
        else:
            st.markdown("Not Found")
    elif os_task_type == "QnA":
        st.markdown(answer)

    if "Chat" in os_task_type:
        if "generated" not in st.session_state:
            st.session_state["generated"] = []

        if "past" not in st.session_state:
            st.session_state["past"] = []

        input_container = st.container()
        colored_header(label="", description="", color_name="blue-30")
        response_container = st.container()

        # Applying the user input box
        with input_container:
            user_input = get_text()

        # Conditional display of AI generated responses as a function of user provided prompts
        with response_container:
            if user_input:
                chat_response = chat_chain(
                    {"input_documents": response["docs"], "human_input": user_input},
                    return_only_outputs=True,
                )
                st.session_state.past.append(user_input)
                st.session_state.generated.append(chat_response["output_text"])

            if st.session_state["generated"]:
                for i in range(len(st.session_state["generated"])):
                    message(
                        st.session_state["past"][i], is_user=True, key=str(i) + "_user"
                    )
                    message(st.session_state["generated"][i], key=str(i))


main()
