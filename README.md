# IMDb Conversational Search App

This repo contains the code to launch a chatbot app that can help users learn more about the movies in IMDb database through conversation search powered by LLMs and LangChain.

Dataset used: MovieLens + IMDb Mojo Dataset

VectorDB: OpenSearch with 100K records

LLM: Anthropic Claude from Amazon Bedrock, Jurassic Jumbo Instruct, Text2Text, Flan T5 XXL

LLM Framework: LangChain

UI Framework: Streamlit

### Features

1. Users have a choice of LLM to select for the conversational search bot: 
![LLM](./img/llm_choice.png)

2. There are two task types that the user can choose from: (Search) or (Search and Chat). First, the LLM searches for movies based on the user query, and then if the chat option is provided, a chat feature is provided to the user. 

The UI provides a set of default questions for the search use case as shown below: 
![default_qns](./img/default_qns.png)

These default as well as user inputted search questions can be grouped as either: 
- Exact match: searching for movies based on location, actor, plot, rating, directors, etc..
- Semantic match: searching for movies that are similar to others 

Currently, the chat bot supports 5 questions at a time to not break context length

![App UI](./img/image.png)


## Instructions to Run Demo

### Set up
1. Run `sh setup.sh` to install dependencies.
2. Process IMDb dataset as per [notebooks/create_datasets](notebooks/create_datasets.ipynb) and [notebooks/IMDB_Dataset_Preprocessing](notebooks/IMDB_Dataset_Preprocessing.ipynb).
3. Create embeddings for IMDb metadata as per [notebooks/embedding_generation](notebooks/embedding_generation.ipynb).
4. Create OpenSearch cluster in the console.  We used the cluster configuration below.
    - instance type = r5.large.search
    - EBS volume size = 30 GiB
    - EBS volume type = General Purpose (SSD) - gp2
    - Number of Availability Zones = 1
    - Ensure the cluster is in closed and private VPC. IMDb is a paid proprietary dataset that SHOULD NOT be exposed to external parties.
5. Create OpenSearch index (exact and semantic match)by running `python src/index_creation.py`. Make sure to configure correct argument values for file paths and index name in the file.


### Bedrock Setup: Optional
In order to follow these instructions, you need access to the private beta Bedrock. 

In terminal: 

1. sh bedrock_install.sh

### Test Streamlit Chat App
1. From folder `streamlit` run `streamlit run chat.py`
2. In another terminal run `bash run.sh` to get the local SM Studio app URL 

### License
This sample code is licensed under the MIT-0 License. See the LICENSE file.