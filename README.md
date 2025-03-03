# JioPay Customer Support RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that automates customer support for JioPay using publicly available information from their website and FAQs.

## Overview

This project implements a conversational AI assistant that can answer questions about JioPay's products and services by retrieving relevant information from a knowledge base built from JioPay's public documentation. The system uses RAG to ensure accurate responses grounded in factual information rather than hallucinated content.

![JioPay Support Chatbot](https://raw.githubusercontent.com/username/jiopay-support-rag/main/assets/chatbot_screenshot.png)

## Table of Contents

- [Data Gathering and Preparation](#data-gathering-and-preparation)
- [Tools and Technologies](#tools-and-technologies)
- [RAG Implementation](#rag-implementation)
- [Advanced Retrieval Techniques](#advanced-retrieval-techniques)
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
   - `improved_jio_pay_scraper.py`: Generic scraper for extracting content from various JioPay webpages including FAQs on multiple pages
   - `run_jiopay_scraper.py`: Orchestration script that runs both scrapers sequentially

2. **Manual Extraction**: Some FAQ content was manually curated to ensure quality and relevance.

### Data Processing

The collected data was processed and stored in JSON format:

- `jiopay_help_center_faqs.json`: Comprehensive FAQs from the Help Center
- `jiopay_links_content.json`: Content from various website pages with metadata

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
   - JSON files are loaded and converted to LangChain Document objects
   - FAQ content is processed differently from general website content
   - Metadata is preserved (source, section, title) for better context and citation

2. **Semantic Document Creation**:
   - **FAQ Documents**: Instead of generic text splitting, we use semantic chunking that preserves the question-answer relationship
   - **Alternative Document Formats**: For each FAQ, we create multiple document representations to enhance retrieval:
     - Standard Q&A format
     - Question-focused format (optimized for direct queries)
     - Section-contextualized format (for topic-based retrieval)
   - **Section Overviews**: Created to help with broad topic questions

3. **Vector Embeddings**:
   - Model: BAAI/bge-base-en-v1.5 (optimized for retrieval tasks)
   - Embeddings are normalized for better similarity calculation
   - Device acceleration used when available (MPS on Apple Silicon)

### Vector Store

- **FAISS Index**: Efficient similarity search for embedding vectors
- Local persistence to disk for reuse across sessions

## Advanced Retrieval Techniques

The system implements several advanced retrieval techniques to improve accuracy and relevance:

### 1. LLM-Based Query Expansion

Rather than using hardcoded heuristics, the system uses the LLM itself to generate better search queries:

- **Query Refinement**: The original user question is sent to the LLM to generate 3-4 alternative formulations
- **Diverse Phrasing**: The LLM creates variations focusing on different aspects and terminology
- **Keyword Extraction**: Technical terms and product names are preserved in the refined queries
- **Fallback Mechanism**: System gracefully falls back to the original query if LLM refinement fails

Benefits:
- Adapts to new product terminology automatically
- Generates semantically related terms humans might miss
- Balances specificity and generality in search

### 2. Semantic Document Chunking

Instead of raw text splitting which breaks question-answer pairs, the system:

- Preserves structured JSON content in semantically meaningful units
- Creates multiple document variants for each FAQ with different formats
- Maintains metadata relationships between questions, answers, and sections
- Includes section context documents for hierarchical understanding
- Generates section overview documents for broad topic questions

### 3. Maximum Marginal Relevance (MMR) Retrieval

To balance relevance with diversity in search results:

- **MMR Algorithm**: Retrieves a larger candidate set, then selects a diverse subset
- **Configurable Parameters**:
  - `k`: Number of documents to return (typically 4-5)
  - `fetch_k`: Initial candidate pool size (typically 10-15)
  - `lambda_mult`: Relevance-diversity tradeoff (0.7 balances both)
- **Fallback to Similarity**: Automatically uses standard similarity search if MMR fails

Benefits:
- Reduces redundancy in retrieval results
- Ensures broader coverage of relevant topics
- Adapts to both specific and general questions

## Project Structure

```
jiopay-support-rag/
├── data/
│   ├── jiopay_help_center_faqs.json
│   └── jiopay_links_content.json
├── scrapers/
│   ├── js_faq_scraper.py
│   ├── improved_jio_pay_scraper.py
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
