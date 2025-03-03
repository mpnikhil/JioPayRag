import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any

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
    def __init__(self):
        # Initialize the embedding model - using LangChain's HuggingFaceEmbeddings for simplicity
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",  # Good quality embeddings
            model_kwargs={"device": "cpu"},      # Use CPU for compatibility (can change to "cuda" or "mps")
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Initialize the LLM using Ollama
        self.llm = ChatOllama(
            model="llama3.3:latest",  # Use default llama3 model (whatever is pulled)
            temperature=0.1  # Low temperature for factual responses
        )
        
        # Initialize the vector store
        self.vectorstore = None
        
        # RAG Chain
        self.qa_chain = None
        
        # Track the last retrieved documents for citation
        self.last_retrieved_docs = []
    
    def load_json_data(self, json_files: List[str]) -> List[Document]:
        """Process JSON files into Documents for the vectorstore."""
        documents = []
        
        for file_path in json_files:
            try:
                # First make sure we're dealing with the correct file path
                # Check if the path contains a directory
                if 'jiopay_data/' in file_path and not os.path.exists(file_path):
                    # Try alternative path without directory
                    alt_path = file_path.replace('jiopay_data/', '')
                    if os.path.exists(alt_path):
                        file_path = alt_path
                    else:
                        print(f"Could not find file at either {file_path} or {alt_path}")
                        continue
                
                print(f"Loading {file_path}...")
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Check data type to handle different JSON structures
                if isinstance(data, str):
                    print(f"Warning: {file_path} contains a string instead of a JSON object")
                    continue
                    
                # Special handling for array of FAQ sections
                if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                    # Check if this looks like the FAQ structure (array of sections with qa_pairs)
                    if any("qa_pairs" in item for item in data):
                        print(f"Processing {file_path} as FAQ structure")
                        for section in data:
                            if not isinstance(section, dict):
                                continue
                                
                            section_name = section.get("section", "General")
                            qa_pairs = section.get("qa_pairs", [])
                            
                            if not isinstance(qa_pairs, list):
                                continue
                                
                            for qa in qa_pairs:
                                if not isinstance(qa, dict):
                                    continue
                                    
                                question = qa.get("question", "")
                                answer = qa.get("answer", "")
                                
                                if question and answer:
                                    doc_content = f"Question: {question}\nAnswer: {answer}"
                                    metadata = {
                                        "source": file_path,
                                        "section": section_name,
                                        "question": question,
                                        "type": "FAQ"
                                    }
                                    documents.append(Document(page_content=doc_content, metadata=metadata))
                    continue  # Skip further processing for this file
                
                # Process website content (links_content.json structure)
                if isinstance(data, dict):
                    # Process home page content
                    if "home_page" in data and isinstance(data["home_page"], dict):
                        home = data["home_page"]
                        for section in home.get("content_sections", []):
                            if not isinstance(section, dict):
                                continue
                                
                            title = section.get("title", "")
                            content = section.get("content", "")
                            
                            if content and len(content) > 20:  # Filter out very short content
                                doc_content = f"{title}\n{content}" if title else content
                                metadata = {
                                    "source": "Home Page",
                                    "section": "General",
                                    "title": title,
                                    "type": "Website Content"
                                }
                                documents.append(Document(page_content=doc_content, metadata=metadata))
                    
                    # Process category content
                    categories = data.get("categories", [])
                    if isinstance(categories, list):
                        for category in categories:
                            if not isinstance(category, dict):
                                continue
                                
                            category_name = category.get("name", "General")
                            links = category.get("links", [])
                            
                            if not isinstance(links, list):
                                continue
                                
                            for link in links:
                                if not isinstance(link, dict):
                                    continue
                                    
                                link_title = link.get("title", "")
                                
                                # Process content sections
                                content_sections = link.get("content_sections", [])
                                if isinstance(content_sections, list):
                                    for section in content_sections:
                                        if not isinstance(section, dict):
                                            continue
                                            
                                        title = section.get("title", "")
                                        content = section.get("content", "")
                                        
                                        if content and len(content) > 20:  # Filter out very short content
                                            doc_content = f"{title}\n{content}" if title else content
                                            metadata = {
                                                "source": link.get("url", ""),
                                                "section": category_name,
                                                "title": link_title,
                                                "type": "Website Content"
                                            }
                                            documents.append(Document(page_content=doc_content, metadata=metadata))
                                
                                # Process FAQs embedded in the website content
                                faqs = link.get("faqs", [])
                                if isinstance(faqs, list):
                                    for faq_section in faqs:
                                        if not isinstance(faq_section, dict):
                                            continue
                                            
                                        section_name = faq_section.get("section", "General")
                                        qa_pairs = faq_section.get("qa_pairs", [])
                                        
                                        if not isinstance(qa_pairs, list):
                                            continue
                                            
                                        for qa in qa_pairs:
                                            if not isinstance(qa, dict):
                                                continue
                                                
                                            question = qa.get("question", "")
                                            answer = qa.get("answer", "")
                                            
                                            if question and answer:
                                                doc_content = f"Question: {question}\nAnswer: {answer}"
                                                metadata = {
                                                    "source": link.get("url", ""),
                                                    "section": section_name,
                                                    "question": question,
                                                    "type": "FAQ"
                                                }
                                                documents.append(Document(page_content=doc_content, metadata=metadata))
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print(f"Successfully loaded {len(documents)} documents from JSON files")
        return documents
    
    def load_csv_data(self, csv_files: List[str]) -> List[Document]:
        """Process CSV files into Documents for the vectorstore."""
        documents = []
        
        for file_path in csv_files:
            try:
                # For FAQ CSV files
                if "faqs" in file_path.lower():
                    df = pd.read_csv(file_path)
                    for _, row in df.iterrows():
                        question = row.get("Question", "")
                        answer = row.get("Answer", "")
                        section = row.get("Section", "")
                        source = row.get("Source", "")
                        
                        if question and answer:
                            doc_content = f"Question: {question}\nAnswer: {answer}"
                            metadata = {
                                "source": file_path,
                                "section": section,
                                "question": question,
                                "original_source": source,
                                "type": "FAQ"
                            }
                            documents.append(Document(page_content=doc_content, metadata=metadata))
                
                # For content CSV files
                elif "content" in file_path.lower():
                    df = pd.read_csv(file_path)
                    for _, row in df.iterrows():
                        category = row.get("Category", "")
                        title = row.get("Link Title", "")
                        url = row.get("URL", "")
                        section_title = row.get("Section Title", "")
                        content = row.get("Content", "")
                        
                        if content and len(content) > 20:  # Filter out very short content
                            doc_content = f"{section_title}\n{content}" if section_title else content
                            metadata = {
                                "source": url or file_path,
                                "section": category,
                                "title": title,
                                "type": "Website Content"
                            }
                            documents.append(Document(page_content=doc_content, metadata=metadata))
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        return documents
    
    def create_knowledge_base(self, json_files=None, csv_files=None):
        """Create and persist the knowledge base from all data sources."""
        documents = []
        
        # Load JSON files if provided
        if json_files:
            documents.extend(self.load_json_data(json_files))
        
        # Load CSV files if provided
        if csv_files:
            documents.extend(self.load_csv_data(csv_files))
        
        if not documents:
            print("Warning: No documents were processed. Check that the data files exist.")
            return None
        
        # Split documents into smaller chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=False
        )
        
        # Process special FAQ content differently - keep QA pairs together
        faq_docs = []
        content_docs = []
        
        for doc in documents:
            if doc.metadata.get("type") == "FAQ":
                faq_docs.append(doc)
            else:
                content_docs.append(doc)
        
        # Split only content documents, keep FAQ documents intact
        split_content_docs = text_splitter.split_documents(content_docs)
        all_docs = faq_docs + split_content_docs
        
        print(f"Processed {len(all_docs)} documents for the knowledge base")
        print(f"- {len(faq_docs)} FAQ documents")
        print(f"- {len(split_content_docs)} content chunks")
        
        # Create and save the FAISS index using LangChain's HuggingFaceEmbeddings
        try:
            self.vectorstore = FAISS.from_documents(
                documents=all_docs,
                embedding=self.embeddings
            )
            
            # Save the vectorstore to disk
            self.vectorstore.save_local("faiss_index")
            print("Knowledge base created and saved to disk")
            
            return self.vectorstore
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            return None
    
    def load_knowledge_base(self, index_path="faiss_index"):
        """Load an existing knowledge base from disk."""
        if os.path.exists(index_path):
            try:
                self.vectorstore = FAISS.load_local(
                    folder_path=index_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"Loaded knowledge base from {index_path}")
                return self.vectorstore
            except Exception as e:
                print(f"Error loading knowledge base: {str(e)}")
                return None
        else:
            print(f"No existing knowledge base found at {index_path}")
            return None
    
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
    
    def build_rag_chain(self):
        """Build the RAG chain with the configured components."""
        # Ensure vectorstore is initialized
        if not self.vectorstore:
            raise ValueError("Vector store is not initialized. Please create or load a knowledge base first.")
        
        # Create a retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Define the prompt template for the RAG chain
        template = """You are a helpful customer support assistant for JioPay, a payment processing service by Jio Payment Solutions Limited in India. 
Your task is to provide accurate and helpful responses to questions about JioPay's services, features, and support processes.

Answer the following question based only on the provided context. Be concise, friendly, and professional in your response.
If the answer cannot be found in the context, politely say you don't have that information and suggest the user contact JioPay support at merchant.support@jiopay.in.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = PromptTemplate.from_template(template)
        
        # Assemble the RAG chain
        self.qa_chain = (
            {"context": retriever | self.format_docs_with_citations, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return self.qa_chain
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Process a user question and return an answer with citations."""
        if not self.qa_chain:
            raise ValueError("QA chain is not initialized. Please build the RAG chain first.")
        
        # Generate the response
        try:
            # Clear previous citations
            self.last_retrieved_docs = []
            
            # Generate response
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

def setup_gradio_interface(rag_system):
    """Set up the Gradio web interface for the chatbot with citations."""
    # Initialize chat history
    chat_history = []
    
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
        chat_history.append((message, formatted_response))
        
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
    
    # Define possible locations for data files
    data_dirs = ["./", "./jiopay_data/", "../jiopay_data/"]
    
    # Base file names
    base_json_files = [
        "jiopay_data/jiopay_help_center_faqs.json",
        "jiopay_data/jiopay_links_content.json"
    ]
    
    base_csv_files = [
        "jiopay_data/jiopay_faqs.csv",
        "jiopay_data/jiopay_links_content.csv"
    ]
    
    # Find existing files across possible directories
    existing_json_files = []
    for base_file in base_json_files:
        found = False
        for data_dir in data_dirs:
            file_path = os.path.join(data_dir, base_file)
            if os.path.exists(file_path):
                existing_json_files.append(file_path)
                found = True
                print(f"Found {base_file} at {file_path}")
                break
        if not found:
            print(f"Warning: Could not find {base_file} in any search path")
    
    existing_csv_files = []
    for base_file in base_csv_files:
        found = False
        for data_dir in data_dirs:
            file_path = os.path.join(data_dir, base_file)
            if os.path.exists(file_path):
                existing_csv_files.append(file_path)
                found = True
                print(f"Found {base_file} at {file_path}")
                break
        if not found:
            print(f"Warning: Could not find {base_file} in any search path")
    
    # Check if knowledge base exists
    if not jiopay_rag.load_knowledge_base():
        print("Creating new knowledge base...")
        if not existing_json_files and not existing_csv_files:
            print("Error: No data files found. Cannot create knowledge base.")
            return
            
        if not jiopay_rag.create_knowledge_base(existing_json_files, existing_csv_files):
            print("Error: Failed to create knowledge base. Exiting.")
            return
    
    # Build the RAG chain
    jiopay_rag.build_rag_chain()
    print("RAG system initialized and ready")
    
    # Launch the Gradio interface
    setup_gradio_interface(jiopay_rag)

if __name__ == "__main__":
    main()