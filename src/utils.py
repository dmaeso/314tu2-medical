import torch

from typing import List, Dict, Any, Union

import numpy as np
from PIL import Image
from io import BytesIO
import base64

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import  UnstructuredURLLoader
from langchain.chains import  RetrievalQA


from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline, 
    GenerationConfig, 
    Pipeline,
    BitsAndBytesConfig
)
from googlesearch import search
import requests

import warnings
warnings.filterwarnings('ignore')





class MistralAIModel:
    def __init__(self, model_name: str) -> None:
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.qc: BitsAndBytesConfig =  BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,

        )
        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=self.qc
        )

        self.generation_config: GenerationConfig = GenerationConfig.from_pretrained(model_name)
        self.generation_config.max_new_tokens = 1024
        self.generation_config.temperature = 0.0001
        self.generation_config.top_p = 0.95
        self.generation_config.do_sample = True
        self.generation_config.repetition_penalty = 1.15

        self.raw_pipeline: Pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=True,
            generation_config=self.generation_config,
        )

        self.llm: HuggingFacePipeline = HuggingFacePipeline(pipeline=self.raw_pipeline)


class DocumentProcessingPipeline:
    def __init__(self, model_name: str, encode_kwargs: Dict, chunk_size: int, chunk_overlap: int, persist_directory: str):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs, model_kwargs={"device": "cuda"})
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.persist_directory = persist_directory

    def process_documents(self, query: str) -> Chroma:
        urls = self._fetch_urls(query)
        loader = UnstructuredURLLoader(urls=urls)
        documents = loader.load()
        text_chunks = self.text_splitter.split_documents(documents)
        return Chroma.from_documents(text_chunks, self.embeddings, persist_directory=self.persist_directory)

    def _fetch_urls(self, query: str) -> List[str]:
        # Placeholder for URL fetching logic
        # Replace this with the actual logic to fetch URLs based on the query
        return [url for url in search(query, lang="en", num_results=10) if 'pdf' not in url]

class RetrievalQAProcessor:
    def __init__(self, qa_chain, captioner):
        self.qa_chain = qa_chain
        self.captioner = captioner

    def image2bytes(self, image_input):
        if isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise TypeError("Input must be an np.ndarray or PIL.Image.Image")
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def extract_answer(self, text):
        return text["result"].split("[/SCENARIO]")[-1].replace("\n","")

    def create_prompt(self, user_input, image_description):
        user_input = user_input if user_input else ''
        image_description = "Image description: " + image_description if image_description else ''
        return f"What am I suffering? {user_input} {image_description}"

    def get_image_description(self, img):
        img_bytes = self.image2bytes(img)
        return self.captioner(img_bytes)[0]['generated_text']

    def retrieval(self, prompt):
        result = self.qa_chain(prompt)
        return self.extract_answer(result)