import logging
from typing import Any
from langchain.chains import  RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
import numpy as np
from PIL import Image
from transformers import (
    pipeline, 
    Pipeline
)

import torch

from src.utils import DocumentProcessingPipeline, MistralAIModel, RetrievalQAProcessor

import gradio as gr

if torch.cuda.is_available():
    print("CUDA enabled! Starting...\n")
else:
    print("CUDA not detected! Exiting...\n")
    exit()
    
def process_inputs(text, image):
    
    if not text and image is None:
        return "No input data"

    # Initialize the QA and captioning processor with the appropriate models and database
    qa_processor = RetrievalQAProcessor(qa_chain=qa_chain, captioner=captioner)

    # Convert image input to the appropriate format if necessary
    if isinstance(image, np.ndarray):  # Image is a NumPy array
        pass  # The image is already in np.ndarray format, no need to convert
    elif isinstance(image, Image.Image):  # Image is a PIL Image
        image = np.array(image)  # Convert PIL Image to np.ndarray
    elif image is not None:
        return "Invalid image format"

    image_description = None
    if image is not None:
        image_description = qa_processor.get_image_description(image)

    prompt = qa_processor.create_prompt(text, image_description)
    return qa_processor.retrieval(prompt)

# Configure logging
logging.basicConfig(level=logging.INFO,filename="app.log", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Initialize and log the AI model setup
model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"
mistral_ai: MistralAIModel = MistralAIModel(model_name)
logging.info(f"Initialized MistralAIModel with model name: {model_name}")
llm: HuggingFacePipeline = mistral_ai.llm

# Initialize and log the document processing pipeline setup
doc_pipeline = DocumentProcessingPipeline(
    model_name="thenlper/gte-large",
    encode_kwargs={"normalize_embeddings": True},
    chunk_size=1024,
    chunk_overlap=64,
    persist_directory="db"
)
logging.info("Initialized DocumentProcessingPipeline")


# Process documents and log the operation
search_query: str = 'first aids guide'
db = doc_pipeline.process_documents(search_query)
logging.info(f"Processed documents for search query: {search_query}")

# Template and prompt setup
template: str  = """[SCENARIO]
Act as an emergency medic. Use the information below to answer the subsequent question.
Include an image description if necessary. For non-health-related queries, respond with an inability to assist on the topic.

[CONTEXT]
{context}

[QUESTION]
{question}
[/SCENARIO]
"""
prompt: PromptTemplate = PromptTemplate(template=template, input_variables=["context", "question"])
logging.info("Template and prompt initialized")

# RetrievalQA setup and logging
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 1}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)
logging.info("RetrievalQA initialized")

# Image captioning pipeline and logging
captioner: Pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
logging.info("Image captioning pipeline initialized")

# Gradio interface setup and launch
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Textbox(label="Introduce your experience or symptoms"),
        gr.Image(label="Drag an image here (optional)"),
    ],
    outputs=gr.Textbox(label="Medical description"),
    title="MedicalQA - First Aids",
    description="As an AI assistant, I'm here to support and provide information. While I can provide a lot of information and support, there are situations where it's crucial to consult a professional."

)
logging.info("Gradio interface initialized")

# Launch the interface
iface.launch(share=True)
logging.info("Gradio interface launched")
