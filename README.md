# JioPay Customer Support RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that automates customer support for JioPay using publicly available information from their website and FAQs.

## Overview

This project implements a conversational AI assistant that can answer questions about JioPay's products and services by retrieving relevant information from a knowledge base built from JioPay's public documentation. The system uses RAG to ensure accurate responses grounded in factual information rather than hallucinated content.

![JioPay Support Chatbot](https://raw.githubusercontent.com/username/jiopay-support-rag/main/assets/chatbot_screenshot.png)

## Table of Contents

- [Data Gathering and Preparation](#data-gathering-and-preparation)
- [Tools and Technologies](#tools-and-technologies)
- [RAG Implementation](#rag-implementation)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Future Enhancements](#future-enhancements)

## Data Gathering and Preparation

### Data Sources

The system uses several data sources from JioPay's public information:

1. **JioPay Help Center FAQs** - Comprehensive FAQs organized by section (JioPay Business App, Dashboard, Collect link, User Management, etc.)
2. **JioPay Website Content** - General information from the JioPay Business website, including product descriptions, features, and benefits
3. **Specialized FAQs** - Domain-specific questions and answers about features like the VoiceBox, Repeat billing, Settlement processes, etc.

### Data Collection Process

Data was collected using two main approaches:

1. **Web Scraping**: Python scripts using Selenium and BeautifulSoup were developed to scrape content from the JioPay Business website:

   - `js_faq_scraper.py`: Specialized scraper for the FAQ section with tailored selectors
   - `simple_jio_pay_scraper.py`: Generic scraper for extracting content from various JioPay webpages
   - `run_jiopay_scraper.py`: Orchestration script that runs both scrapers sequentially

2. **Manual Extraction**: Some FAQ content was manually curated to ensure quality and relevance.

### Data Processing

The collected data was processed and stored in multiple formats:

- **JSON Files**: 
  - `jiopay_help_center_faqs.json`: Comprehensive FAQs from the Help Center
  - `jiopay_links_content.json`: Content from various website pages with metadata

- **CSV Files**: 
  - `jiopay_faqs.csv`: Tabular format of all FAQs with source, section, question, and answer
  - `jiopay_links_content.csv`: General content from the website in a tabular format

## Tools and Technologies

### Core Technologies

- **Python 3.9+**: Primary programming language
- **LangChain**: Framework for building LLM-powered applications
- **FAISS**: Vector database for efficient similarity search
- **HuggingFace Embeddings**: Using BAAI/bge-base-en-v1.5 for high-quality embeddings
- **Ollama**: Local LLM serving for inference with Llama 3
- **Gradio**: Web interface for the chatbot

### Key Dependencies

```
langchain
langchain_core
langchain_text_splitters
langchain_ollama
faiss-cpu
sentence-transformers
huggingface_hub
numpy
pandas
gradio
```

### Web Scraping Tools

- **Selenium**: Browser automation for dynamic content
- **BeautifulSoup**: HTML parsing and content extraction
- **Chrome WebDriver**: Headless browser for scraping

## RAG Implementation

### Knowledge Base Construction

The knowledge base is constructed through the following process:

1. **Document Processing**:
   - JSON and CSV files are loaded and converted to LangChain Document objects
   - FAQ content is processed differently from general website content
   - Metadata is preserved (source, section, title) for better context and citation

2. **Chunking Strategy**:
   - **FAQ Documents**: Kept intact to preserve question-answer pairs
   - **General Content**: Split using RecursiveCharacterTextSplitter with:
     - Chunk size: 1000 characters
     - Chunk overlap: 200 characters
     - Custom separators: paragraphs, sentences, spaces

3. **Vector Embeddings**:
   - Model: BAAI/bge-base-en-v1.5 (optimized for retrieval tasks)
   - Embeddings are normalized for better similarity calculation
   - Device acceleration used when available (MPS on Apple Silicon)

4. **Vector Store**:
   - FAISS index for fast similarity search
   - Local persistence to disk for reuse across sessions

### Retrieval Process

The retrieval process is optimized for JioPay customer support:

1. **Query Processing**:
   - User question is embedded using the same embedding model
   - Semantic search in the vector space

2. **Document Retrieval**:
   - Top-k retrieval (k=5) based on similarity scores
   - Emphasis on FAQ content as they are kept intact

3. **Context Formation**:
   - Retrieved documents are concatenated
   - Full context is passed to the LLM

### Language Model Integration

The system uses a locally-served LLM through Ollama:

1. **Model**: Llama 3 (llama3.3:latest)
2. **Temperature**: 0.1 (low temperature for factual responses)
3. **Custom Prompt Template**:
   ```
   You are a helpful customer support assistant for JioPay.
   Answer the following question based only on the provided context.
   If the answer cannot be found in the context, suggest contacting JioPay support.
   
   Context:
   {context}
   
   Question: {question}
   
   Answer:
   ```

## Project Structure

```
jiopay-support-rag/
├── data/
│   ├── faq_data.json
│   ├── jiopay_help_center_faqs.json
│   ├── jiopay_links_content.json
│   ├── jiopay_faqs.csv
│   └── jiopay_links_content.csv
├── scrapers/
│   ├── js_faq_scraper.py
│   ├── simple_jio_pay_scraper.py
│   └── run_jiopay_scraper.py
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
├── jiopay_support_rag.py
├── requirements.txt
└── README.md
```

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/jiopay-support-rag.git
   cd jiopay-support-rag
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama**:
   - Follow instructions at [ollama.ai](https://ollama.ai)
   - Pull the Llama 3 model:
     ```bash
     ollama pull llama3.3:latest
     ```

5. **Run the application**:
   ```bash
   python jiopay_support_rag.py
   ```

## Usage

1. Once launched, the application will start a Gradio web interface at http://localhost:7860
2. Type your JioPay-related questions in the chat interface
3. The system will retrieve relevant information and generate a response
4. Sample questions are provided as examples in the interface

## Future Enhancements

- **Multilingual Support**: Add support for Indian regional languages
- **Hybrid Search**: Combine vector similarity with keyword-based search
- **Response Citations**: Add direct links to source documents
- **Conversation History**: Enhance with memory of previous interactions
- **Feedback Loop**: Add user feedback mechanism to improve responses