#!/usr/bin/env python3
"""
Vietnamese Legal Chatbot - Setup Script
This script initializes the RAG system and processes the legal documents.
"""

import os
import sys
from typing import Dict, Any
from utils.data_loader import LegalDataLoader
from main.chatbot import VietnameseLegalRAG
from config import Config

def check_environment():
    """Check if all required environment variables are set"""
    print("Checking environment configuration...")
    
    missing_vars = []
    
    if not Config.GOOGLE_API_KEY:
        missing_vars.append("GOOGLE_API_KEY")
    
    if not Config.QDRANT_URL:
        missing_vars.append("QDRANT_URL") 
    
    if not Config.QDRANT_API_KEY:
        missing_vars.append("QDRANT_API_KEY")
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("\nPlease set the following environment variables:")
        for var in missing_vars:
            print(f"  export {var}=your_value_here")
        print("\nOr create a .env file with these variables.")
        return False
    
    print("✅ Environment configuration OK")
    return True

def check_data_files():
    """Check if required data files exist"""
    print("Checking data files...")
    
    required_files = [
        Config.CORPUS_PATH,
        Config.STOPWORDS_PATH,
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing data files: {', '.join(missing_files)}")
        return False
    
    print("✅ Data files OK")
    return True

def setup_rag_system(force_rebuild: bool = False):
    """Setup the RAG system with indices"""
    print("Setting up RAG system...")
    
    try:
        # Initialize data loader
        print("Initializing data loader...")
        data_loader = LegalDataLoader()
        
        # Load legal documents
        print("Loading legal corpus...")
        legal_docs = data_loader.load_legal_corpus()
        
        if not legal_docs:
            print("❌ No legal documents loaded")
            return None
        
        # Prepare documents for indexing
        print("Preparing documents for indexing...")
        documents = data_loader.prepare_documents_for_indexing()
        
        if not documents:
            print("❌ No documents prepared for indexing")
            return None
        
        print(f"📚 Prepared {len(documents)} documents for indexing")
        
        # Initialize RAG system with component-by-component error handling
        print("Initializing RAG system components...")
        
        try:
            print("  - Initializing text processor...")
            from utils.text_processor import VietnameseTextProcessor
            text_processor = VietnameseTextProcessor()
            print("  ✅ Text processor initialized")
        except Exception as e:
            print(f"  ❌ Text processor error: {e}")
            raise
        
        try:
            print("  - Initializing vector store...")
            from main.vector_store import QdrantVectorStore
            vector_store = QdrantVectorStore()
            print("  ✅ Vector store initialized")
        except Exception as e:
            print(f"  ❌ Vector store error: {e}")
            raise
        
        try:
            print("  - Initializing BM25 retriever...")
            from main.bm25_retriever import BM25Retriever
            bm25_retriever = BM25Retriever()
            print("  ✅ BM25 retriever initialized")
        except Exception as e:
            print(f"  ❌ BM25 retriever error: {e}")
            raise
        
        try:
            print("  - Initializing complete RAG system...")
            rag_system = VietnameseLegalRAG()
            print("  ✅ RAG system initialized")
        except Exception as e:
            print(f"  ❌ RAG system initialization error: {e}")
            raise
        
        # Setup indices
        print("Building indices (this may take a while)...")
        rag_system.setup_indices(documents, force_rebuild=force_rebuild)
        
        print("✅ RAG system setup completed")
        return rag_system
        
    except UnicodeDecodeError as e:
        print(f"❌ Encoding error setting up RAG system: {e}")
        print("💡 Try running: python cleanup.py")
        print("💡 Then run setup again: python setup_system.py")
        return None
    except Exception as e:
        print(f"❌ Error setting up RAG system: {e}")
        print("💡 For encoding issues, try: python cleanup.py")
        import traceback
        print("Full error traceback:")
        traceback.print_exc()
        return None

def test_system(rag_system):
    """Test the RAG system with sample questions"""
    print("\nTesting RAG system...")
    
    test_questions = [
        "Quyền và nghĩa vụ của người lao động là gì?",
        "Thời gian làm việc theo quy định của pháp luật?",
        "Điều kiện kết hôn theo luật hôn nhân và gia đình?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Test {i}: {question} ---")
        
        try:
            result = rag_system.answer_question(question, use_fallback=False)
            
            print(f"Answer: {result['answer'][:200]}...")
            print(f"Retrieved docs: {len(result['retrieved_documents'])}")
            print(f"Fallback used: {result['fallback_used']}")
            
        except Exception as e:
            print(f"Error answering question: {e}")

def display_system_status(rag_system):
    """Display system status and statistics"""
    print("\n" + "="*50)
    print("SYSTEM STATUS")
    print("="*50)
    
    status = rag_system.get_system_status()
    
    print(f"🤖 LLM Available: {'✅' if status['llm_available'] else '❌'}")
    print(f"🔍 Vector Store: {'✅' if status['vector_store_available'] else '❌'}")
    print(f"📊 BM25 Retriever: {'✅' if status['bm25_available'] else '❌'}")
    print(f"🔑 Google API: {'✅' if status['google_api_configured'] else '❌'}")
    print(f"☁️  QDrant Cloud: {'✅' if status['qdrant_configured'] else '❌'}")
    
    # Vector store info
    if 'vector_store_info' in status and status['vector_store_info']:
        info = status['vector_store_info']
        print(f"\n📚 Vector Store Info:")
        print(f"  - Collection: {info.get('name', 'N/A')}")
        print(f"  - Documents: {info.get('points_count', 0):,}")
        vectors_count = info.get('vectors_count')
        print(f"  - Vectors: {vectors_count if vectors_count is not None else 0:,}")
    
    # BM25 stats
    if 'bm25_stats' in status and status['bm25_stats']:
        stats = status['bm25_stats']
        print(f"\n📊 BM25 Index Stats:")
        print(f"  - Documents: {stats.get('total_documents', 0):,}")
        print(f"  - Vocabulary: {stats.get('vocabulary_size', 0):,}")
        print(f"  - Avg Doc Length: {stats.get('average_document_length', 0):.1f}")

def main():
    """Main setup function"""
    print("🏛️  Vietnamese Legal Chatbot - Setup")
    print("="*50)
    
    # Check prerequisites
    if not check_environment():
        print("\n❌ Environment check failed. Please configure your environment variables.")
        sys.exit(1)
    
    if not check_data_files():
        print("\n❌ Data file check failed. Please ensure all data files are present.")
        sys.exit(1)
    
    # Parse command line arguments
    force_rebuild = "--rebuild" in sys.argv or "-r" in sys.argv
    run_tests = "--test" in sys.argv or "-t" in sys.argv
    
    if force_rebuild:
        print("\n🔄 Force rebuild mode enabled")
    
    # Setup RAG system
    rag_system = setup_rag_system(force_rebuild=force_rebuild)
    
    if not rag_system:
        print("\n❌ RAG system setup failed")
        sys.exit(1)
    
    # Display system status
    display_system_status(rag_system)
    
    # Run tests if requested
    if run_tests:
        test_system(rag_system)
    
    print("\n✅ Setup completed successfully!")
    print("\nYou can now run the Streamlit app:")
    print("  streamlit run app.py")
    
    print("\nUsage:")
    print("  python setup_system.py           # Normal setup")
    print("  python setup_system.py --rebuild # Force rebuild indices")
    print("  python setup_system.py --test    # Run with tests")

if __name__ == "__main__":
    main() 