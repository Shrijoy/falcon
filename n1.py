import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_huggingface import HuggingFaceEndpoint
import os
# from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()

# os.environ["HF_TOKEN"] = os.getenv("hugging_write")

api_key = os.getenv("hugging_write")

# loader=PyPDFLoader("transformer.pdf")
# docs=loader.load()

template ="""
Give a summary of the document.
document : {text}
Summary : """

final_prompt = """Provide the final complete summary of the document.
document : {text}"""

pdf = st.file_uploader("Upload the pdf file",type="pdf", accept_multiple_files= False)
# pdf = PyPDFLoader("transformer.pdf")

if pdf:
    temppdf = f"./temp.pdf"
    with open(temppdf,"wb") as file:
        file.write(pdf.getvalue())
        file_name=pdf.name

    loader=PyPDFLoader(temppdf)
    docs=loader.load()

    # docs = fetch_pdf(pdf) 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)
    final_document = text_splitter.split_documents(docs)

    temperature = 0.1
    max_tokens = 1500
    # hf_api_key = os.getenv("hugging_face_token")
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = HuggingFaceEndpoint(repo_id="tiiuae/falcon-7b-instruct",temperature=temperature,token=api_key)
    prompt = PromptTemplate(input_variables=['text'],template=template)
    final_prompt_template = PromptTemplate(input_variable=['text'], template=final_prompt)
    chain = load_summarize_chain(llm,chain_type="map_reduce", map_prompt=prompt, combine_prompt=final_prompt_template)
    response = chain.run(final_document)
    st.write("Summary: ", response)
   

# from langchain_huggingface import HuggingFaceEndpoint
# repo_id = "mistralai/Mistral-7B-Instruct-v0.2"  
# llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=10, temperature=0.1, token=api_key)

# response = llm.invoke("what is AI?")

# print(response)