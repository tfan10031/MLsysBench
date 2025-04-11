from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
import os

# Load the API keys from a JSON file.
with open('keys.json', 'r') as file:
    keys = json.load(file)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'C:/Users/tiany/Desktop/MLsysBench/filter/mlsysbench-751ba36b50d8.json'

# Define the Pydantic model with correct field names.
class DomainInfo(BaseModel):
    isDomainSpecific: bool = Field(description="Whether this pull request requires hardware-specific knowledge except Nvidia CUDA")
    device: str = Field(description="Device type related to this pull request in small letter")
    reason: str = Field(description="Reason for the decision")

# Create a parser that will expect output matching DomainInfo.
parser = PydanticOutputParser(pydantic_object=DomainInfo)

# Build the prompt with explicit formatting instructions.
prompt = PromptTemplate(
    template=(
        "You are a GitHub pull request reviewer.\n\n"
        "Please analyze the following pull request information and decide whether it requires hardware-specific knowledge "
        "(for example, knowledge about AMD ROCm, Apple MPS, etc.).\n\n"
        "However, if the required hardware-specific knowledge is only about Nvidia CUDA, you should return False.\n\n"
        "Follow these formatting instructions EXACTLY: {format_instructions}\n\n"
        "Pull Request Information:\n"
        "Patch: {patch}\n"
        "Problem Statement: {problem_statement}\n"
        "Hints: {hints}\n\n"
        "Provide your analysis as a valid python dictionary without any additional markdown formatting."
    ),
    input_variables=["patch", "problem_statement", "hints"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Create the chain by combining the prompt, the LLM, and the parser.
chain_openai = prompt | ChatOpenAI(
    model="gpt-4o-2024-08-06", 
    temperature=0.0, 
    openai_api_key=keys['openai']
) | parser

chain_groq = prompt | ChatGroq(
    model='llama-3.3-70b-versatile', 
    temperature=0.0, 
    api_key=keys['groq']
) | parser

chain_vertexai = prompt | ChatVertexAI(
    model='gemini-2.0-flash',
    temperature=0.0
) | parser



def isDomainSpecific(instance: dict, service: str) -> dict:
    """
    This function evaluates whether a GitHub pull request requires hardware-specific expertise.
    It invokes the language model chain and parses the response into a DomainInfo object.
    """
    if service == 'openai':
        chain = chain_openai
    elif service == 'groq':
        chain = chain_groq
    elif service == 'vertexai':
        chain = chain_vertexai
    else:
        raise ValueError(f"Service not supported: {service}")
    try:
        # Call the chain with the provided pull request details.
        result = chain.invoke({
            'patch': instance['patch'],
            'problem_statement': instance['problem_statement'],
            'hints': instance['hints_text']
        })
        # Return the result using the exact field names from the DomainInfo model.
        return {
            'isDomainSpecific': result.isDomainSpecific,
            'device': result.device,
            'reason': result.reason
        }
    except Exception as e:
        # In case of a parsing or evaluation error, return error details.
        return {
            'isDomainSpecific': None,
            'device': None,
            'reason': f"Analysis failed: {str(e)}"
        }
