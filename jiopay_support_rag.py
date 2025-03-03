import os
import json
import numpy as np
import pandas as pd
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

class JioPaySupportRAG:
    """
    A Retrieval-Augmented Generation system for JioPay customer support.
    This class handles loading data, creating a knowledge base, and answering questions.
    """
    
    def __init__(self, model_name: str = "llama3.3:latest", embedding_model: str = "BAAI/bge-base-en-v1.5"):
        """
        Initialize the RAG system with the specified models.
        
        Args:
            model_name: The name of the Ollama model to use
            embedding_model: The name of the HuggingFace embedding model to use
        """
        # Initialize the embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},  # Change to "cuda" for GPU or "mps" for M1/M2 Mac
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Initialize the LLM using Ollama
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1  # Low temperature for factual responses
        )
        
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
    
    def load_faq_data(self, faq_file_path: str) -> List[Document]:
        """
        Load FAQ data from a JSON file.
        
        Args:
            faq_file_path: Path to the FAQ JSON file
            
        Returns:
            List of Document objects for the knowledge base
        """
        documents = []
        
        try:
            with open(faq_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for section in data:
                section_name = section.get("section", "General")
                for qa in section.get("qa_pairs", []):
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
        
        except Exception as e:
            print(f"Error processing FAQ file {faq_file_path}: {str(e)}")
        
        return documents
    
    def load_site_content(self, site_content_file_path: str) -> List[Document]:
        """
        Load website content data from a JSON file.
        
        Args:
            site_content_file_path: Path to the site content JSON file
            
        Returns:
            List of Document objects for the knowledge base
        """
        documents = []
        
        try:
            with open(site_content_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Process home page content
            if "home_page" in data:
                home = data["home_page"]
                url = home.get("url", "")
                
                # Combine all relevant home page content into one document
                home_content = ""
                home_title = home.get("title", "Home")
                
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
            
            # Process categories and their links
            for category in data.get("categories", []):
                category_name = category.get("name", "General")
                
                for link in category.get("links", []):
                    link_title = link.get("title", "")
                    url = link.get("url", "")
                    
                    # Group related content sections for each link
                    link_content = f"Page: {link_title}\n"
                    
                    # Process content sections
                    for section in link.get("content_sections", []):
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
                    for faq_section in link.get("faqs", []):
                        section_name = faq_section.get("section", "General")
                        for qa in faq_section.get("qa_pairs", []):
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
        
        except Exception as e:
            print(f"Error processing site content file {site_content_file_path}: {str(e)}")
        
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
        documents = []
        
        # Load FAQ data
        if faq_files:
            for faq_file in faq_files:
                if os.path.exists(faq_file):
                    faq_docs = self.load_faq_data(faq_file)
                    documents.extend(faq_docs)
                    print(f"Loaded {len(faq_docs)} FAQ documents from {faq_file}")
                else:
                    print(f"Warning: FAQ file {faq_file} not found")
        
        # Load site content data
        if site_content_file and os.path.exists(site_content_file):
            content_docs = self.load_site_content(site_content_file)
            documents.extend(content_docs)
            print(f"Loaded {len(content_docs)} content documents from {site_content_file}")
        elif site_content_file:
            print(f"Warning: Site content file {site_content_file} not found")
        
        if not documents:
            print("Error: No documents were loaded. Check your file paths.")
            return False
        
        # Create and save the FAISS index
        try:
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(index_path) if os.path.dirname(index_path) else ".", exist_ok=True)
            
            # Save the vectorstore to disk
            self.vectorstore.save_local(index_path)
            print(f"Knowledge base created with {len(documents)} documents and saved to {index_path}")
            return True
            
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            return False
    
    def load_knowledge_base(self, index_path: str = "faiss_index") -> bool:
        """
        Load an existing knowledge base from disk.
        
        Args:
            index_path: Path to the saved FAISS index
            
        Returns:
            bool: True if successful, False otherwise
        """
        if os.path.exists(index_path):
            try:
                self.vectorstore = FAISS.load_local(
                    folder_path=index_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"Loaded knowledge base from {index_path}")
                return True
            except Exception as e:
                print(f"Error loading knowledge base: {str(e)}")
                return False
        else:
            print(f"No existing knowledge base found at {index_path}")
            return False
    
    def format_docs_with_citations(self, docs):
        """Format documents and store them for citation purposes."""
        # Store the documents for citation generation
        self.last_retrieved_docs = docs
        
        # Format for the context
        return "\n\n".join([doc.page_content for doc in docs])
    
    def get_citations(self):
        """Generate citations from the retrieved documents."""
        citations = []
        seen_citations = set()  # Track seen citations to avoid duplicates
        
        for doc in self.last_retrieved_docs:
            # Extract citation info from the document
            metadata = doc.metadata
            doc_type = metadata.get("type", "Documentation")
            
            if doc_type == "FAQ":
                question = metadata.get("question", "")
                section = metadata.get("section", "General")
                
                # Create citation string
                citation = f"FAQ: {question} (Section: {section})"
                
                # Only add if not seen before
                if citation not in seen_citations:
                    citations.append(citation)
                    seen_citations.add(citation)
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
        
        return citations

    def build_rag_chain(self, k: int = 5):
        """
        Build the RAG chain with the configured components.
        
        Args:
            k: Number of documents to retrieve for each query
        """
        # Ensure vectorstore is initialized
        if not self.vectorstore:
            raise ValueError("Vector store is not initialized. Please create or load a knowledge base first.")
        
        # Create a retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        # Create the prompt
        prompt = PromptTemplate.from_template(self.prompt_template)
        
        # Define the formatting function for context
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        
        # Assemble the RAG chain
        self.qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return self.qa_chain
    
    def answer_question(self, question: str) -> str:
        """
        Process a user question and return an answer.
        
        Args:
            question: The user's question
            
        Returns:
            str: The generated answer
        """
        # Clear previous citations
        self.last_retrieved_docs = []
        if not self.qa_chain:
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
                break
        
        # Generate the response
        try:
            # If we have a filter, use it
            if search_filter and self.vectorstore:
                filtered_retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5, "filter": search_filter}
                )
                
                # Format documents function
                def format_docs(docs):
                        """Format documents and store them for citation purposes."""
                        # Store the documents for citation generation
                        self.last_retrieved_docs = docs
        
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
                response = temp_chain.invoke(question)
            else:
                # Use the default chain
                response = self.qa_chain.invoke(question)
                
            # Generate citations
            citations = self.get_citations()
            
            return {
                "answer": response,
                "citations": citations
            }
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return {
                "answer": "I'm sorry, I encountered an error while processing your question. Please try again.",
                "citations": []
            }
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I'm sorry, I encountered an error while processing your question. Please try again."

def setup_gradio_interface(rag_system):
    """Set up the Gradio web interface for the chatbot with citations."""
    
    def respond(message, history):
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
    demo.launch(
        share=True,  # Creates a public URL through tunneling
        server_name="0.0.0.0",  # Binds to all network interfaces
        server_port=7860,  # Default Gradio port
        allowed_paths=["faiss_index"],  # Allow access to the FAISS index directory
    )

def main():
    """Main function to initialize and run the JioPay Support RAG chatbot."""
    # Initialize the RAG system
    jiopay_rag = JioPaySupportRAG()
    
    faq_files = ["jiopay_data/jiopay_help_center_faqs.json"]
    site_content_file = "jiopay_data/jiopay_links_content.json"
      
    # Check if knowledge base exists
    if not jiopay_rag.load_knowledge_base():
        print("Creating new knowledge base...")
        if not faq_files and not site_content_file:
            print("Error: No data files found. Cannot create knowledge base.")
            return
            
        if not jiopay_rag.create_knowledge_base(
            faq_files=faq_files,
            site_content_file=site_content_file
        ):
            print("Error: Failed to create knowledge base. Exiting.")
            return
    
    # Build the RAG chain
    jiopay_rag.build_rag_chain()
    print("RAG system initialized and ready")
    
    # Launch the Gradio interface
    setup_gradio_interface(jiopay_rag)
    #print(jiopay_rag.answer_question(" What is Jiopay"))


if __name__ == "__main__":
    main()