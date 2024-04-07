# Use an official Python 3.10 image as the base
FROM python:3.10

RUN pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

# Install the required Python libraries
RUN pip install gradio \
    xformer==1.0.1 \
    chromadb==0.4.24 \
    langchain==0.1.14 \
    accelerate==0.29.1 \
    transformers==4.38.2 \
    bitsandbytes==0.43.0 \
    unstructured==0.13.2 \
    sentence-transformers==2.6.1 \
    googlesearch-python==1.2.3 \
    pysqlite3-binary

# Set the working directory in the container to /app
WORKDIR /app

# Copy the Gradio app source code to the container
COPY . /app

# Expose the port the app runs on
EXPOSE 7860

# Command to run the Gradio app
CMD ["python", "app.py"]
