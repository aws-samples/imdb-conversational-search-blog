import sagemaker
from sagemaker import ModelPackage

from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings, SentenceTransformerEmbeddings 
from langchain import HuggingFaceHub, SagemakerEndpoint

import boto3
import os
import yaml

from src.content_handlers import SageMakerContentHandler, AI21SageMakerContentHandler

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)


def text2text_llm():
    """
    Create a text to text llm directly from HuggingFace
    Return:
        Langchain LLM (HuggingFace)
    """
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = config["HUGGINGFACEHUB_API_TOKEN"]
    # prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-xl", model_kwargs={"temperature": 1e-10}
    )
    return llm

def launch_encoder(model_name="gtr-t5-large"):
    """
    Launch sentence transformer embedding model
    Args:
        model_name(str): specific sentence transformer model
    Returns:
        SentenceTransformerEmbeddings: embedding model
    """
    return SentenceTransformerEmbeddings(model_name=model_name)

def amazon_bedrock_embeddings():
    """
    Launch bedrock embedding model
    Return:
        langchain.embeddings.BedrockEmbeddings: bedrock embedding model
    """
    return BedrockEmbeddings()

def create_bedrock_body(temp = 0.0, topP = 1, stop_sequences=[]):
    """
    Configurations for the bedrock llm model
    Args:
        temp(float): variability of model results
        topP(integer): control how deterministic the model is
        stop_sequences(list): stop generating text at these specific words
    """
    body = {
        "max_tokens_to_sample": 300,
        "temperature": temp,
        "top_k": 250,
        "top_p":topP,
        "stop_sequences":stop_sequences   
       } 
    return body


def amazon_bedrock_llm(
    region="us-east-1", modelId='anthropic.claude-v1', verbose=False
):
    """
    Create bedrock llm from langchain. Make sure to have bedrock capabilities in this account
    Args:
        region(str): AWS region
        modelId(str): model type
    Returns:
        langchain.llms.Bedrock: bedrock llm
    """
    
    llm = Bedrock(region_name=region, model_id=modelId)
    llm.client = boto3.client(
        service_name="bedrock",
        region_name=region,
        endpoint_url=f"https://bedrock.{region}.amazonaws.com",
    )
    llm.model_kwargs = create_bedrock_body()
    return llm


def sagemaker_endpoint(endpoint_name):
    """
    Create langchain llm from a T5-XXL Endpoint within SM
    Args:
        endpoint_name(str): name of the J2 Jumbo Instruct endpoint
    Returns:
        Langchain llm
    """
    return SagemakerEndpoint(
        endpoint_name=endpoint_name,
        region_name="us-east-1",
        model_kwargs={"temperature": 1e-10},
        content_handler=SageMakerContentHandler(),
    )


def sagemaker_endpoint_ai21(endpoint_name):
    """
    Create langchain llm from a J2 Jumbo Instruct Endpoint within SM
    Args:
        endpoint_name(str): name of the J2 Jumbo Instruct endpoint
    Returns:
        Langchain llm
    """
    return SagemakerEndpoint(
        endpoint_name=endpoint_name,
        region_name="us-east-1",
        model_kwargs={"temperature": 0, "maxTokens": 300, "numResult": 1},
        content_handler=AI21SageMakerContentHandler(),
    )


def create_endpoint_AI21(
    endpoint_name="j2-jumbo-instruct",
    content_type="application/json",
    real_time_inference_instance_type="ml.g5.48xlarge",
):
    """
    Run this script to create a j2-jumbo-instruct endpoint within sagemaker.
    Args:
        endpoint_name(str): name of the AI21 LLM endpoint
        content_type(str): input and output types of the endpoint
        real_time_inference_instance_type(str): instance type to use for the endpoint
    """
    model_package_map = {
        "us-east-1": "arn:aws:sagemaker:us-east-1:865070037744:model-package/j2-jumbo-instruct-v1-1-033-87b797db88313edf9c3851adf6fc371f",
        "us-east-2": "arn:aws:sagemaker:us-east-2:057799348421:model-package/j2-jumbo-instruct-v1-1-033-87b797db88313edf9c3851adf6fc371f",
        "us-west-1": "arn:aws:sagemaker:us-west-1:382657785993:model-package/j2-jumbo-instruct-v1-1-033-87b797db88313edf9c3851adf6fc371f",
        "us-west-2": "arn:aws:sagemaker:us-west-2:594846645681:model-package/j2-jumbo-instruct-v1-1-033-87b797db88313edf9c3851adf6fc371f",
        "ca-central-1": "arn:aws:sagemaker:ca-central-1:470592106596:model-package/j2-jumbo-instruct-v1-1-033-87b797db88313edf9c3851adf6fc371f",
        "eu-central-1": "arn:aws:sagemaker:eu-central-1:446921602837:model-package/j2-jumbo-instruct-v1-1-033-87b797db88313edf9c3851adf6fc371f",
        "eu-west-1": "arn:aws:sagemaker:eu-west-1:985815980388:model-package/j2-jumbo-instruct-v1-1-033-87b797db88313edf9c3851adf6fc371f",
        "eu-west-2": "arn:aws:sagemaker:eu-west-2:856760150666:model-package/j2-jumbo-instruct-v1-1-033-87b797db88313edf9c3851adf6fc371f",
        "eu-west-3": "arn:aws:sagemaker:eu-west-3:843114510376:model-package/j2-jumbo-instruct-v1-1-033-87b797db88313edf9c3851adf6fc371f",
        "eu-north-1": "arn:aws:sagemaker:eu-north-1:136758871317:model-package/j2-jumbo-instruct-v1-1-033-87b797db88313edf9c3851adf6fc371f",
        "ap-southeast-1": "arn:aws:sagemaker:ap-southeast-1:192199979996:model-package/j2-jumbo-instruct-v1-1-033-87b797db88313edf9c3851adf6fc371f",
        "ap-southeast-2": "arn:aws:sagemaker:ap-southeast-2:666831318237:model-package/j2-jumbo-instruct-v1-1-033-87b797db88313edf9c3851adf6fc371f",
        "ap-northeast-2": "arn:aws:sagemaker:ap-northeast-2:745090734665:model-package/j2-jumbo-instruct-v1-1-033-87b797db88313edf9c3851adf6fc371f",
        "ap-northeast-1": "arn:aws:sagemaker:ap-northeast-1:977537786026:model-package/j2-jumbo-instruct-v1-1-033-87b797db88313edf9c3851adf6fc371f",
        "ap-south-1": "arn:aws:sagemaker:ap-south-1:077584701553:model-package/j2-jumbo-instruct-v1-1-033-87b797db88313edf9c3851adf6fc371f",
        "sa-east-1": "arn:aws:sagemaker:sa-east-1:270155090741:model-package/j2-jumbo-instruct-v1-1-033-87b797db88313edf9c3851adf6fc371f",
    }

    region = boto3.Session().region_name
    if region not in model_package_map.keys():
        raise ("UNSUPPORTED REGION")

    model_package_arn = model_package_map[region]

    role = sagemaker.get_execution_role()
    sagemaker_session = sagemaker.Session()

    # create a deployable model from the model package.
    model = ModelPackage(
        role=role,
        model_package_arn=model_package_arn,
        sagemaker_session=sagemaker_session,
    )

    # Deploy the model
    model.deploy(
        1,
        real_time_inference_instance_type,
        endpoint_name=endpoint_name,
        model_data_download_timeout=3600,
        container_startup_health_check_timeout=600,
    )
