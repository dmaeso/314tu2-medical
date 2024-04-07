
# Medical Assistant Application

## Overview

This Medical Assistant Application utilizes advanced Large Language Models (LLMs) to provide insights and assistance in medical-related inquiries and tasks. It's designed to leverage powerful computational resources to deliver high accuracy and performance.

## System Requirements

### Hardware Requirements

- **GPU:** A GPU device with at least 16 GB of GPU memory is required to efficiently run the application (CUDA). This is crucial for handling the computational demands of the LLMs.

### Software Requirements

- **Docker:** The application is containerized using Docker, ensuring consistency across different environments. Make sure Docker is installed and running on your system.

## Installation & Execution (local)

1. **Cloning the Repository:**


   ```bash
   git clone https://gitfront.io/r/314tu2/qLaDsrvrKe1e/314tu2-medical.git
   cd 314tu2-medical

2. **Building the Docker Image:**


   ```bash
   sudo docker build -t medical .
   ```

3. **Running the Docker Container:**

   Once the Docker image is built, run the container using the following command:

   ```bash
   docker run --gpus all -p 7860:7860 medical
   ```
## Installation & Execution (Google Collaborate)


1. **Access to Google Collaborate**


    [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmaeso/314tu2-medical/blob/main/MedicalAssistant_Demo.ipynb)


2. **Ensure GPU usage: Runtime Environment > Change runtime type > Hardware accelerator > T4 GPU**

3. **Execute cell (Gradio link will be shown)**


