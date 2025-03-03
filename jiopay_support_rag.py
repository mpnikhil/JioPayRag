import os
import json
import numpy as np
import pandas as pd
import time
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

# RAG components
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Web UI
import gradio as gr

# Set environment variables for reproducibility
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("jiopay_rag_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("JioPayRAG")

class JioPaySupportRAG:
    """
    A Retrieval-Augmented Generation system for JioPay customer support.
    This class handles loading data, creating a knowledge base, and answering questions.
    """
    
    def __init__(self, model_name: str = "llama3.3:latest", embedding_model: str = "BAAI/bge-base-en-v1.5", debug: bool = False):
        """
        Initialize the RAG system with the specified models.
        
        Args:
            model_name: The name of the Ollama model to use
            embedding_model: The name of the HuggingFace embedding model to use
            debug: Whether to enable debug mode with verbose logging
        """
        self.debug = debug
        self.start_time = time.time()
        self.query_count = 0
        
        logger.info(f"Initializing JioPaySupportRAG with model={model_name}, embeddings={embedding_model}")
        
        # Initialize the embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": "cpu"},  # Change to "cuda" for GPU or "mps" for M1/M2 Mac
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info(f"Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
        
        # Initialize the LLM using Ollama
        logger.info(f"Loading LLM: {model_name}")
        try:
            self.llm = ChatOllama(
                model=model_name,
                temperature=0.1  # Low temperature for factual responses
            )
            logger.info(f"LLM loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM: {str(e)}")
            raise
        
        # Initialize the vector store
        self.vectorstore = None
        
        # RAG Chain
        self.qa_chain = None
        
        # Define the prompt template
        self.prompt_template = """You are a helpful customer support assistant for JioPay, a payment processing service by Jio Payment Solutions Limited in India. 
Your task is to provide accurate and helpful responses to questions about JioPay's services, features, and support processes.

Answer the following question based only on the provided context. Be concise, friendly, and professional in your response.
If the answer cannot be found in the context, politely say you don't have that information and suggest the user contact JioPay support at merchant.support@jiopay.in.

If the question is about a technical issue or troubleshooting, provide step-by-step instructions.
If the question is about a feature or service, explain how it works and its benefits.
If the question is about a process (like refunds or settlements), clearly outline the required steps.

Context:
{context}

Question: {question}

Answer:"""

        logger.info(f"JioPaySupportRAG initialization completed in {time.time() - self.start_time:.2f} seconds")
    
    def load_faq_data(self, faq_file_path: str) -> List[Document]:
        """
        Load FAQ data from a JSON file.
        
        Args:
            faq_file_path: Path to the FAQ JSON file
            
        Returns:
            List of Document objects for the knowledge base
        """
        start_time = time.time()
        logger.info(f"Loading FAQ data from: {faq_file_path}")
        documents = []
        
        try:
            with open(faq_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            logger.info(f"Loaded JSON with {len(data)} sections")
            
            total_qa_pairs = 0
            for section in data:
                section_name = section.get("section", "General")
                qa_pairs = section.get("qa_pairs", [])
                total_qa_pairs += len(qa_pairs)
                
                if self.debug:
                    logger.debug(f"Processing section: {section_name} with {len(qa_pairs)} QA pairs")
                
                for qa in qa_pairs:
                    question = qa.get("question", "")
                    answer = qa.get("answer", "")
                    
                    if question and answer:
                        # Format with clear structure for better retrieval
                        doc_content = f"Question: {question}\nAnswer: {answer}"
                        metadata = {
                            "source": faq_file_path,
                            "section": section_name,
                            "question": question,
                            "type": "faq"
                        }
                        documents.append(Document(page_content=doc_content, metadata=metadata))
            
            processing_time = time.time() - start_time
            logger.info(f"Processed {total_qa_pairs} QA pairs and created {len(documents)} documents in {processing_time:.2f} seconds")
        
        except Exception as e:
            logger.error(f"Error processing FAQ file {faq_file_path}: {str(e)}")
            if self.debug:
                import traceback
                logger.debug(traceback.format_exc())
        
        return documents
    
    def load_site_content(self, site_content_file_path: str) -> List[Document]:
        """
        Load website content data from a JSON file.
        
        Args:
            site_content_file_path: Path to the site content JSON file
            
        Returns:
            List of Document objects for the knowledge base
        """
        start_time = time.time()
        logger.info(f"Loading site content from: {site_content_file_path}")
        documents = []
        
        try:
            with open(site_content_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            logger.info(f"Loaded site content JSON successfully")
            
            # Count metrics for logging
            home_sections = 0
            categories_count = 0
            links_count = 0
            content_sections_count = 0
            faq_sections_count = 0
            embedded_qa_count = 0
            
            # Process home page content
            if "home_page" in data:
                home = data["home_page"]
                url = home.get("url", "")
                
                # Combine all relevant home page content into one document
                home_content = ""
                home_title = home.get("title", "Home")
                home_sections = len(home.get("content_sections", []))
                
                if self.debug:
                    logger.debug(f"Processing home page with {home_sections} content sections")
                
                for section in home.get("content_sections", []):
                    title = section.get("title", "")
                    content = section.get("content", "")
                    
                    if content and len(content) > 20:  # Filter out very short content
                        if title:
                            home_content += f"{title}: {content}\n\n"
                        else:
                            home_content += f"{content}\n\n"
                
                if home_content:
                    metadata = {
                        "source": url,
                        "section": "Home",
                        "title": home_title,
                        "type": "content"
                    }
                    documents.append(Document(page_content=home_content, metadata=metadata))
                    if self.debug:
                        logger.debug(f"Added home page document with {len(home_content)} chars")
            
            # Process categories and their links
            categories_count = len(data.get("categories", []))
            if self.debug:
                logger.debug(f"Processing {categories_count} categories")
                
            for category in data.get("categories", []):
                category_name = category.get("name", "General")
                category_links = category.get("links", [])
                links_count += len(category_links)
                
                if self.debug:
                    logger.debug(f"Processing category: {category_name} with {len(category_links)} links")
                
                for link in category_links:
                    link_title = link.get("title", "")
                    url = link.get("url", "")
                    link_sections = link.get("content_sections", [])
                    content_sections_count += len(link_sections)
                    
                    if self.debug and link_title:
                        logger.debug(f"Processing link: {link_title} with {len(link_sections)} content sections")
                    
                    # Group related content sections for each link
                    link_content = f"Page: {link_title}\n"
                    
                    # Process content sections
                    for section in link_sections:
                        title = section.get("title", "")
                        content = section.get("content", "")
                        
                        if content and len(content) > 20:  # Filter out very short content
                            section_content = f"Section: {title}\n{content}\n\n" if title else content
                            
                            # For longer sections, create a separate document
                            if len(content) > 500:
                                section_metadata = {
                                    "source": url,
                                    "section": category_name,
                                    "title": link_title,
                                    "subsection": title,
                                    "type": "content"
                                }
                                section_doc = f"Page: {link_title}\nSection: {title}\n\n{content}"
                                documents.append(Document(page_content=section_doc, metadata=section_metadata))
                            else:
                                # For shorter sections, add to the combined link content
                                link_content += section_content
                    
                    # Add the combined link content as a document if it's not just the title
                    if len(link_content) > len(f"Page: {link_title}\n"):
                        link_metadata = {
                            "source": url,
                            "section": category_name,
                            "title": link_title,
                            "type": "content"
                        }
                        documents.append(Document(page_content=link_content, metadata=link_metadata))
                    
                    # Process FAQs embedded in the website content (these are already structured)
                    link_faqs = link.get("faqs", [])
                    faq_sections_count += len(link_faqs)
                    
                    for faq_section in link_faqs:
                        section_name = faq_section.get("section", "General")
                        qa_pairs = faq_section.get("qa_pairs", [])
                        embedded_qa_count += len(qa_pairs)
                        
                        if self.debug:
                            logger.debug(f"Processing embedded FAQ section: {section_name} with {len(qa_pairs)} QA pairs")
                        
                        for qa in qa_pairs:
                            question = qa.get("question", "")
                            answer = qa.get("answer", "")
                            
                            if question and answer:
                                doc_content = f"Question: {question}\nAnswer: {answer}"
                                metadata = {
                                    "source": url,
                                    "section": section_name,
                                    "page": link_title,
                                    "question": question,
                                    "type": "faq"
                                }
                                documents.append(Document(page_content=doc_content, metadata=metadata))
            
            processing_time = time.time() - start_time
            logger.info(f"Site content processing stats:")
            logger.info(f"- Home sections: {home_sections}")
            logger.info(f"- Categories: {categories_count}")
            logger.info(f"- Links: {links_count}")
            logger.info(f"- Content sections: {content_sections_count}")
            logger.info(f"- FAQ sections: {faq_sections_count}")
            logger.info(f"- Embedded QA pairs: {embedded_qa_count}")
            logger.info(f"- Total documents created: {len(documents)}")
            logger.info(f"- Processing time: {processing_time:.2f} seconds")
        
        except Exception as e:
            logger.error(f"Error processing site content file {site_content_file_path}: {str(e)}")
            if self.debug:
                import traceback
                logger.debug(traceback.format_exc())
        
        return documents
    
    def create_knowledge_base(self, 
                             faq_files: Optional[List[str]] = None, 
                             site_content_file: Optional[str] = None,
                             index_path: str = "faiss_index") -> bool:
        """
        Create and save a knowledge base from the provided data files.
        
        Args:
            faq_files: List of paths to FAQ JSON files
            site_content_file: Path to the site content JSON file
            index_path: Path to save the FAISS index
            
        Returns:
            bool: True if successful, False otherwise
        """
        start_time = time.time()
        logger.info(f"Creating knowledge base from {len(faq_files) if faq_files else 0} FAQ files and site content")
        documents = []
        
        # Load FAQ data
        if faq_files:
            for faq_file in faq_files:
                if os.path.exists(faq_file):
                    faq_docs = self.load_faq_data(faq_file)
                    documents.extend(faq_docs)
                    logger.info(f"Loaded {len(faq_docs)} FAQ documents from {faq_file}")
                else:
                    logger.warning(f"FAQ file {faq_file} not found")
        
        # Load site content data
        if site_content_file and os.path.exists(site_content_file):
            content_docs = self.load_site_content(site_content_file)
            documents.extend(content_docs)
            logger.info(f"Loaded {len(content_docs)} content documents from {site_content_file}")
        elif site_content_file:
            logger.warning(f"Site content file {site_content_file} not found")
        
        if not documents:
            logger.error("No documents were loaded. Check your file paths.")
            return False
        
        # Statistics about documents
        faq_count = sum(1 for doc in documents if doc.metadata.get("type") == "faq")
        content_count = sum(1 for doc in documents if doc.metadata.get("type") == "content")
        avg_doc_length = sum(len(doc.page_content) for doc in documents) / len(documents)
        
        logger.info(f"Document statistics:")
        logger.info(f"- Total documents: {len(documents)}")
        logger.info(f"- FAQ documents: {faq_count}")
        logger.info(f"- Content documents: {content_count}")
        logger.info(f"- Average document length: {avg_doc_length:.2f} characters")
        
        # Create and save the FAISS index
        try:
            logger.info("Creating vector embeddings - this may take some time...")
            embedding_start = time.time()
            
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            embedding_time = time.time() - embedding_start
            logger.info(f"Vector embeddings created in {embedding_time:.2f} seconds")
            
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(index_path) if os.path.dirname(index_path) else ".", exist_ok=True)
            
            # Save the vectorstore to disk
            save_start = time.time()
            logger.info(f"Saving vector store to {index_path}...")
            self.vectorstore.save_local(index_path)
            save_time = time.time() - save_start
            
            total_time = time.time() - start_time
            logger.info(f"Knowledge base created and saved successfully:")
            logger.info(f"- Embedding time: {embedding_time:.2f} seconds")
            logger.info(f"- Save time: {save_time:.2f} seconds")
            logger.info(f"- Total time: {total_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            if self.debug:
                import traceback
                logger.debug(traceback.format_exc())
            return False
    
    def load_knowledge_base(self, index_path: str = "faiss_index") -> bool:
        """
        Load an existing knowledge base from disk.
        
        Args:
            index_path: Path to the saved FAISS index
            
        Returns:
            bool: True if successful, False otherwise
        """
        start_time = time.time()
        logger.info(f"Loading knowledge base from {index_path}")
        
        if os.path.exists(index_path):
            try:
                self.vectorstore = FAISS.load_local(
                    folder_path=index_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                load_time = time.time() - start_time
                
                # Try to get index stats
                try:
                    index_size = self.vectorstore.index.ntotal
                    logger.info(f"Knowledge base loaded with {index_size} vectors in {load_time:.2f} seconds")
                except:
                    logger.info(f"Knowledge base loaded in {load_time:.2f} seconds")
                
                return True
            except Exception as e:
                logger.error(f"Error loading knowledge base: {str(e)}")
                if self.debug:
                    import traceback
                    logger.debug(traceback.format_exc())
                return False
        else:
            logger.warning(f"No existing knowledge base found at {index_path}")
            return False
    
    def format_docs_with_citations(self, docs):
        """Format documents and store them for citation purposes."""
        if self.debug:
            logger.debug(f"Retrieved {len(docs)} documents for context")
            for i, doc in enumerate(docs):
                doc_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                logger.debug(f"Doc {i+1}: {doc_preview}")
                logger.debug(f"Metadata: {doc.metadata}")
        
        # Store the documents for citation generation
        self.last_retrieved_docs = docs
        
        # Format for the context
        return "\n\n".join([doc.page_content for doc in docs])
    
    def get_citations(self):
        """Generate citations from the retrieved documents."""
        citations = []
        seen_citations = set()  # Track seen citations to avoid duplicates
        
        if self.debug:
            logger.debug(f"Generating citations from {len(self.last_retrieved_docs)} documents")
        
        for doc in self.last_retrieved_docs:
            # Extract citation info from the document
            metadata = doc.metadata
            doc_type = metadata.get("type", "Documentation")
            
            if doc_type == "FAQ" or doc_type == "faq":
                question = metadata.get("question", "")
                section = metadata.get("section", "General")
                
                # Create citation string
                citation = f"FAQ: {question} (Section: {section})"
                
                # Only add if not seen before
                if citation not in seen_citations:
                    citations.append(citation)
                    seen_citations.add(citation)
                    if self.debug:
                        logger.debug(f"Added FAQ citation: {citation}")
            else:
                title = metadata.get("title", "")
                section = metadata.get("section", "")
                source = metadata.get("source", "").replace("https://jiopay.com/business/", "")
                
                # Create citation string
                citation = f"{title} ({source})"
                
                # Only add if not seen before and not empty
                if citation not in seen_citations and title:
                    citations.append(citation)
                    seen_citations.add(citation)
                    if self.debug:
                        logger.debug(f"Added content citation: {citation}")
        
        return citations

    def build_rag_chain(self, k: int = 5):
        """
        Build the RAG chain with the configured components.
        
        Args:
            k: Number of documents to retrieve for each query
        """
        # Ensure vectorstore is initialized
        if not self.vectorstore:
            logger.error("Vector store is not initialized. Please create or load a knowledge base first.")
            raise ValueError("Vector store is not initialized. Please create or load a knowledge base first.")
        
        logger.info(f"Building RAG chain with k={k} documents per query")
        start_time = time.time()
        
        # Create a retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        # Create the prompt
        prompt = PromptTemplate.from_template(self.prompt_template)
        
        # Define the formatting function for context
        def format_docs(docs):
            # Log the retrieved documents if in debug mode
            if self.debug:
                logger.debug(f"Retrieved {len(docs)} documents for context")
                for i, doc in enumerate(docs):
                    logger.debug(f"Document {i+1} metadata: {doc.metadata}")
            
            # Store documents for citation generation
            self.last_retrieved_docs = docs
            
            return "\n\n".join([doc.page_content for doc in docs])
        
        # Assemble the RAG chain
        self.qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        build_time = time.time() - start_time
        logger.info(f"RAG chain built in {build_time:.2f} seconds")
        
        return self.qa_chain
    
    def answer_question(self, question: str) -> str:
        """
        Process a user question and return an answer.
        
        Args:
            question: The user's question
            
        Returns:
            str: The generated answer
        """
        self.query_count += 1
        start_time = time.time()
        
        logger.info(f"Query #{self.query_count}: \"{question}\"")
        
        # Clear previous citations
        self.last_retrieved_docs = []
        if not self.qa_chain:
            logger.error("QA chain is not initialized. Please build the RAG chain first.")
            raise ValueError("QA chain is not initialized. Please build the RAG chain first.")
        
        # Detect question type to improve retrieval
        question_lower = question.lower()
        search_filter = None
        
        # Use metadata filters based on question type if appropriate
        filter_terms = {
            "Refunds": ["refund", "money back", "return payment", "cancel payment"],
            "Settlement": ["settlement", "receive money", "funds", "deposit"],
            "Voicebox": ["voice", "voicebox", "speaker", "audio", "sound"],
            "JioPay Business App": ["app", "mobile app", "download app", "android app"],
            "Campaign": ["campaign", "promotion", "offer", "discount"],
            "Collect link": ["collect link", "payment link", "generate link"]
        }
        
        # Check if question matches any filter terms
        for section, terms in filter_terms.items():
            if any(term in question_lower for term in terms):
                search_filter = {"section": section}
                logger.info(f"Applied filter for section: {section}")
                break
        
        # Generate the response
        try:
            retrieval_start = time.time()
            
            # If we have a filter, use it
            if search_filter and self.vectorstore:
                if self.debug:
                    logger.debug(f"Using filtered retriever with filter: {search_filter}")
                
                filtered_retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5, "filter": search_filter}
                )
                
                # Format documents function
                def format_docs(docs):
                    """Format documents and store them for citation purposes."""
                    # Store the documents for citation generation
                    self.last_retrieved_docs = docs
                    
                    if self.debug:
                        logger.debug(f"Retrieved {len(docs)} documents with filter")
                        for i, doc in enumerate(docs):
                            logger.debug(f"Doc {i+1} - {doc.metadata.get('type')} - {doc.metadata.get('section')}")
                    
                    # Format for the context
                    return "\n\n".join([doc.page_content for doc in docs])
                
                # Create a temporary chain with the filtered retriever
                temp_chain = (
                    {"context": filtered_retriever | format_docs, "question": RunnablePassthrough()}
                    | PromptTemplate.from_template(self.prompt_template)
                    | self.llm
                    | StrOutputParser()
                )
                
                # Use the filtered chain
                retrieval_time = time.time() - retrieval_start
                logger.info(f"Retrieval completed in {retrieval_time:.2f} seconds")
                
                generation_start = time.time()
                response = temp_chain.invoke(question)
                generation_time = time.time() - generation_start
                
            else:
                # Use the default chain
                if self.debug:
                    logger.debug("Using default retriever without filters")
                
                retrieval_time = time.time() - retrieval_start
                logger.info(f"Retrieval completed in {retrieval_time:.2f} seconds")
                
                generation_start = time.time()
                response = self.qa_chain.invoke(question)
                generation_time = time.time() - generation_start
                
            # Generate citations
            citations = self.get_citations()
            
            total_time = time.time() - start_time
            logger.info(f"Response generated in {total_time:.2f} seconds")
            logger.info(f"- Retrieval: {retrieval_time:.2f}s, Generation: {generation_time:.2f}s")
            logger.info(f"- Used {len(self.last_retrieved_docs)} docs, Generated {len(citations)} citations")
            
            # Log a preview of the response
            response_preview = response[:100] + "..." if len(response) > 100 else response
            logger.info(f"Response: {response_preview}")
            
            return {
                "answer": response,
                "citations": citations
            }
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"Error generating response after {error_time:.2f} seconds: {str(e)}")
            if self.debug:
                import traceback
                logger.debug(traceback.format_exc())
            
            return {
                "answer": "I'm sorry, I encountered an error while processing your question. Please try again.",
                "citations": []
            }

def setup_gradio_interface(rag_system):
    """Set up the Gradio web interface for the chatbot with citations."""
    logger.info("Setting up Gradio interface")
    
    def respond(message, history):
        # Log the user query
        logger.info(f"User query: {message}")
        query_start = time.time()
        
        # Process the user's question
        response_data = rag_system.answer_question(message)
        
        answer = response_data["answer"]
        citations = response_data["citations"]
        
        # Format the response with citations
        if citations:
            formatted_response = f"{answer}\n\n**Sources:**\n"
            for i, citation in enumerate(citations[:3], 1):  # Limit to top 3 citations
                formatted_response += f"{i}. {citation}\n"
        else:
            formatted_response = answer
        
        query_time = time.time() - query_start
        logger.info(f"Total response time: {query_time:.2f} seconds")
        
        # Update history
        history.append((message, formatted_response))
        
        return formatted_response
    
    # Create the Gradio interface
    demo = gr.ChatInterface(
        fn=respond,
        title="JioPay Customer Support Assistant",
        description="""Ask questions about JioPay's payment services, features, app usage, 
settlement processes, refunds, and more. This assistant is powered by JioPay's official 
documentation and FAQs.""",
        examples=[
            "What is JioPay Business?",
            "How can I issue refunds to my customers?",
            "What are the payment modes available via Collect link?",
            "How can I download the JioPay Business app?",
            "What if my settlement is on hold?",
            "How does the VoiceBox work?",
            "Can I add sub-users to my JioPay Business account?"
        ],
        theme="soft"
    )
    
    # Configure CORS for public access (tunneling)
    logger.info("Launching Gradio interface")
    demo.launch(
        share=True,  # Creates a public URL through tunneling
        server_name="0.0.0.0",  # Binds to all network interfaces
        server_port=7860,  # Default Gradio port
        allowed_paths=["faiss_index"],  # Allow access to the FAISS index directory
    )

def main():
    """Main function to initialize and run the JioPay Support RAG chatbot."""
    start_time = time.time()
    logger.info("Starting JioPay Support RAG system")
    
    # Check for debug flag in environment
    debug_mode = os.environ.get("JIOPAY_DEBUG", "false").lower() == "true"
    logger.info(f"Debug mode: {'enabled' if debug_mode else 'disabled'}")
    
    # Initialize the RAG system
    try:
        logger.info("Initializing RAG system...")
        jiopay_rag = JioPaySupportRAG(debug=debug_mode)
        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Set up data file paths
    faq_files = ["jiopay_data/jiopay_help_center_faqs.json"]
    site_content_file = "jiopay_data/jiopay_links_content.json"
    
    # Log data file paths
    logger.info(f"FAQ files: {faq_files}")
    logger.info(f"Site content file: {site_content_file}")
    
    # Validate data files exist
    missing_files = []
    for faq_file in faq_files:
        if not os.path.exists(faq_file):
            missing_files.append(faq_file)
    
    if site_content_file and not os.path.exists(site_content_file):
        missing_files.append(site_content_file)
    
    if missing_files:
        logger.warning(f"Missing data files: {missing_files}")
      
    # Check if knowledge base exists
    kb_load_start = time.time()
    if os.path.exists("faiss_index"):
        logger.info("Existing knowledge base found, attempting to load...")
        kb_loaded = jiopay_rag.load_knowledge_base()
        kb_load_time = time.time() - kb_load_start
        
        if kb_loaded:
            logger.info(f"Successfully loaded existing knowledge base in {kb_load_time:.2f} seconds")
        else:
            logger.warning(f"Failed to load existing knowledge base in {kb_load_time:.2f} seconds")
            
            if not faq_files and not site_content_file:
                logger.error("No data files found. Cannot create knowledge base.")
                return
                
            logger.info("Creating new knowledge base...")
            kb_create_start = time.time()
            kb_created = jiopay_rag.create_knowledge_base(
                faq_files=faq_files,
                site_content_file=site_content_file
            )
            kb_create_time = time.time() - kb_create_start
            
            if not kb_created:
                logger.error(f"Failed to create knowledge base in {kb_create_time:.2f} seconds. Exiting.")
                return
            logger.info(f"Successfully created new knowledge base in {kb_create_time:.2f} seconds")
    else:
        logger.info("No existing knowledge base found")
        
        if not faq_files and not site_content_file:
            logger.error("No data files found. Cannot create knowledge base.")
            return
            
        logger.info("Creating new knowledge base...")
        kb_create_start = time.time()
        kb_created = jiopay_rag.create_knowledge_base(
            faq_files=faq_files,
            site_content_file=site_content_file
        )
        kb_create_time = time.time() - kb_create_start
        
        if not kb_created:
            logger.error(f"Failed to create knowledge base in {kb_create_time:.2f} seconds. Exiting.")
            return
        logger.info(f"Successfully created new knowledge base in {kb_create_time:.2f} seconds")
    
    # Build the RAG chain
    chain_build_start = time.time()
    try:
        logger.info("Building RAG chain...")
        jiopay_rag.build_rag_chain()
        chain_build_time = time.time() - chain_build_start
        logger.info(f"RAG chain built successfully in {chain_build_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Failed to build RAG chain: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Calculate total initialization time
    init_time = time.time() - start_time
    logger.info(f"Total initialization time: {init_time:.2f} seconds")
    logger.info("RAG system initialized and ready")
    
    # Launch the Gradio interface
    try:
        logger.info("Starting Gradio web interface...")
        setup_gradio_interface(jiopay_rag)
    except Exception as e:
        logger.error(f"Error launching Gradio interface: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

# Add a command-line execution option with debug flag
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="JioPay Support RAG Chatbot")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        default="INFO", help="Set the logging level")
    
    args = parser.parse_args()
    
    # Configure logging based on arguments
    if args.debug:
        os.environ["JIOPAY_DEBUG"] = "true"
        logging.getLogger("JioPayRAG").setLevel(logging.DEBUG)
    else:
        logging.getLogger("JioPayRAG").setLevel(getattr(logging, args.log_level))
    
    # Record start time for performance metrics
    script_start = time.time()
    logger.info(f"Starting JioPay RAG script at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        script_runtime = time.time() - script_start
        logger.info(f"Script completed in {script_runtime:.2f} seconds")