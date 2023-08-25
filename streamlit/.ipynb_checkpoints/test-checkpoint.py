from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
import boto3

def amazon_bedrock_embeddings(modelId="amazon.titan-tg1-large"):
    return BedrockEmbeddings(model_id=modelId)


def amazon_bedrock_llm(
    region="us-east-1", modelId="amazon.titan-tg1-large", verbose=False
):
    """
    Create bedrock LLM from boto3 api
    
    bedrock = boto3.client(
        service_name="bedrock",
        region_name=region,
        endpoint_url=f"https://bedrock.{region}.amazonaws.com",
    )
    resp = bedrock.invoke_model(modelId=modelId , body=json.dumps(body), accept = "*/*", contentType = "application/json") # "amazon.titan-tg1-large"
    """
    """
    Create bedrock llm from langchain
    """
    
    llm = Bedrock(region_name=region, model_id=modelId)
    llm.client = boto3.client(
        service_name="bedrock",
        region_name=region,
        endpoint_url=f"https://bedrock.{region}.amazonaws.com",
    )
    return llm

llm = amazon_bedrock_llm()

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

conversation.predict(input="Hi there!")

