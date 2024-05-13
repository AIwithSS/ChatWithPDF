#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain import document_loaders as dl
from langchain.llms import HuggingFacePipeline
from langchain.chains.summarize import load_summarize_chain
import torch


# In[ ]:


# Load LLM Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained("zephyr-7b-beta-tokenizer")
model = AutoModelForCausalLM.from_pretrained("zephyr-7b-beta-model", torch_dtype=torch.float32)


# In[ ]:


# For CPU only configuration, change accordingly for CUDA
pipe = pipeline(task="text-generation", model=model,tokenizer=tokenizer, device="cpu", max_new_tokens=1000)


# In[ ]:


llm=HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':0})


# In[ ]:


def summary():
    document_path = input("Please provide the PDF file to summarize: ")
    loader = dl.PyPDFLoader(document_path)
    document = loader.load()
    # Define prompt
    prompt_template = """Write a summary of the following in 4000 words:
    "{text}"
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    print(stuff_chain.run(document))


# In[ ]:





# In[ ]:




