"""
UI format of the IMDB demo including choices given to the user for the LLM to use, that task to perform, questions to ask, and many more. 
"""
import requests
import yaml

import sys
import base64

sys.path.append("..")

from src.prompt import CHAT_PROMPT
from src.search import initialize_ops
import src.langchain_qa_chat as langchain_qa_chat
import src.endpoint as endpoint

import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header

with open("../config.yml", "r") as file:
    config = yaml.safe_load(file)

st.set_page_config(layout="wide")


@st.cache_resource
def check_env():
    langchain_qa_chat.validate_environment()


@st.cache_data
def list_llm_models():
    """
    Available models.
    """
    return [
        "Bedrock",
        "Jurassic-Jumbo-Instruct" 
    ]


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
    """
    Default questions in the UI
    """
    return [
        "What are the movies starring Tom Cruise?",
        "What are the movies directed by James Cameron and rating greater than 5.5?",
        "What are the movies starring Kate Winslet and starring Leonardo DiCaprio?",
        "What movies are trending today?",
        "Ask your question",
    ]


@st.cache_resource
def create_qa_chain(model, os_task_type, verbose=False):
    """
    Create a question answering chain from langchain using an llm chosen by the user.
    Args:
        model(str): llm model type
        os_task_type(str): check if it wants a chat functionality
        verbose(boolean): whether to have the model show its full output
    Returns:
        Jurassic Jumbo/Bedrock: llm model used for search and chat
        langchain.chains.question_answering : QA chain for chat using LLM
        OpenSearch: initialized opensearch instance
        SentenceTransformer: sentence transformer embedding model
    """
    if model == "Bedrock":
        llm = endpoint.amazon_bedrock_llm(verbose=verbose)
    elif model == "Jurassic-Jumbo-Instruct":
        llm = endpoint.sagemaker_endpoint_ai21(config["llm"]["ai21_instruct"])
    else:
        assert False
    embedding_model = endpoint.launch_encoder()
    print("Make QA chain for", model)
    chat_chain = None
    if "Chat" in os_task_type:
        chat_chain = langchain_qa_chat.chain_chat(
            llm, verbose=verbose, prompt=CHAT_PROMPT
        )

    ops = initialize_ops()

    return llm, chat_chain, ops, embedding_model


def get_text():
    """
    Function for taking user provided prompt as input
    Returns:
        str: text input
    """
    input_text = st.text_input("You: ", "", key="input")
    return input_text

# def submit():
#     st.session_state.past.append(st.session_state.input)
#     st.session_state.input = ""

def on_btn_click():
    del st.session_state.past[:]
    del st.session_state.generated[:]
    st.session_state.input = ""

def main():
    check_env()

    col_title = st.columns(3)
    col_title[1].title("IMDb Conversational Search")

    col_model = st.columns(3)
    model = col_model[1].selectbox("Select LLM", list_llm_models())

    os_task_type = st.columns(3)[1].selectbox("Task Type", list_task_type())

    col_query = st.columns(3)
    query = col_query[1].selectbox("Select question", [""] + list_questions())
    if "Ask" in query:
        query = st.columns(3)[1].text_input(
            "Your Question: ", placeholder="Ask me anything ...", key="input_qn"
        )
    if not query:
        return

    k = recommended_k()
    answer = "Not found"
    print("Q:", query)
    store = "CUSTOM"

    for attempt in range(1):
        try:
            search_llm, chat_chain, ops, embedding_model = create_qa_chain(
                model, os_task_type, verbose=True
            )

            response = langchain_qa_chat.search_and_answer(
                store,
                search_llm,
                query,
                ops,
                embedding_model,
                k=k,
                task=os_task_type,
            )
            answer = response["response"]
            break  # Success
        except Exception as e:
            st.spinner(text=type(e).__name__)
            if type(e).__name__ == "ValidationException" and k > 1:
                print("Retrying using shorter context")
                k -= 1
            elif type(e).__name__ == "ThrottlingException":
                print("Retrying")
            else:
                # continue
                raise e

    # Setup search interface
    if "Search" in os_task_type:
        if len(answer) > 0:
            cols = st.columns(10)
            for i, (title, poster, ttid, trailer_url) in enumerate(answer[0:10]):
                if "trend" in query and trailer_url != "No link available":
                    url = trailer_url
                else:
                    url = f"https://www.imdb.com/title/{ttid}/"
                # print(imdb_url)
                try:
                    image_html = get_img_with_href(poster, url)
                    # print(image_html)
                    cols[i % 10].markdown(image_html, unsafe_allow_html=True)
                except requests.exceptions.MissingSchema:
                    cols[i % 10].markdown("")
                cols[i % 10].markdown(f"[{title}]({url})")
                if i % 10 == 0 and i > 1:
                    cols = st.columns(10)
        else:
            st.markdown("Not Found")
    elif os_task_type == "QnA":
        st.markdown(answer)

    # Setup chat interface
    if "Chat" in os_task_type:
        if "generated" not in st.session_state:
            st.session_state["generated"] = ["Ask any question about the movies above!"]

        if "past" not in st.session_state:
            st.session_state["past"] = ["User text here"]

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
                    
                st.button("Clear message", on_click=on_btn_click)


main()
