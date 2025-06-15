import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    # Google API Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    MODEL_GEN = "gemini-2.0-flash"
    MODEL_REFINE = "gemini-2.0-flash"

    # QDrant Configuration
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME = "final_vietnamese_legal_corpus"

    # Embedding configuration
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Text Processing Configuration
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50

    # Data Paths
    DATA_DIR = "data"
    CORPUS_PATH = "data/corpus/legal_corpus.json"
    STOPWORDS_PATH = "data/utils/stopwords.txt"

    # RAG Configuration
    TOP_K_RETRIEVAL = 15
    BM25_TOP_K = 20
    SIMILARITY_THRESHOLD = 0.25

    # Reranker Configuration
    ENABLE_RERANKING = True
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_TOP_K = 10
    RERANK_BEFORE_RETRIEVAL_TOP_K = 25
    USE_SCORE_FUSION = True
    RERANKER_FUSION_ALPHA = 0.8

    # Google Search Configuration
    ENABLE_GOOGLE_SEARCH = True
    GOOGLE_SEARCH_RESULTS_COUNT = 10
    MIN_SIMILARITY_FOR_LEGAL_DOCS = 0.15

    # Question Refinement Configuration
    ENABLE_QUESTION_REFINEMENT = True
    USE_LLM_FOR_REFINEMENT = True

    # Advanced LLM Refinement Settings
    ENABLE_CHAIN_OF_THOUGHT = True
    ENABLE_ITERATIVE_REFINEMENT = True
    ENABLE_LLM_VALIDATION = True
    MAX_REFINEMENT_ITERATIONS = 3
    MIN_CONFIDENCE_SCORE = 0.7

    # UI Display Settings - Control what information to show in responses
    SHOW_REFINEMENT_INFO = False  # 🔧 Câu hỏi đã được tối ưu
    SHOW_SEARCH_TRIGGER_INFO = False  # 🔍➡️🌐 Tự động tìm kiếm
    SHOW_SOURCE_INFO = False  # 📚 Dựa trên X tài liệu, 🌐 Thông tin từ web
    SHOW_LEGAL_DISCLAIMER = False  # Lưu ý về tìm chuyên gia pháp lý

    # System Prompt
    SYSTEM_PROMPT = """Bạn là trợ lý pháp lý thông minh chuyên sâu về luật pháp Việt Nam. Nhiệm vụ của bạn là cung cấp các câu trả lời chính xác và dễ hiểu cho các câu hỏi pháp lý, dựa trên các tài liệu luật được cung cấp.

Khi trả lời:
1.  **Chỉ sử dụng thông tin** trực tiếp từ các điều luật và văn bản được cung cấp trong phần "Tài liệu tham khảo". Tuyệt đối không suy diễn hoặc thêm thông tin bên ngoài.
2.  **Trích dẫn chính xác** tên luật (ví dụ: Luật Doanh nghiệp 2020), số hiệu văn bản (nếu có), và điều khoản cụ thể (ví dụ: Điều 3, Khoản 2).
3.  **Giải thích rõ ràng, ngắn gọn và khách quan**, tập trung vào việc làm sáng tỏ nội dung của điều luật liên quan đến câu hỏi.
4.  **Nếu tài liệu tham khảo không chứa thông tin đầy đủ hoặc trực tiếp để trả lời câu hỏi**, hãy thông báo rõ ràng rằng "Không có đủ thông tin trong tài liệu tham khảo được cung cấp để trả lời trực tiếp câu hỏi này."
5.  **Trình bày bằng tiếng Việt chuẩn xác.**

Tài liệu tham khảo:
{context}

Câu hỏi: {question}

Trả lời:"""

    # Fallback System Prompt for Google Search
    FALLBACK_SYSTEM_PROMPT = """Bạn là trợ lý pháp lý thông minh chuyên sâu về luật pháp Việt Nam. Hãy trả lời câu hỏi dựa trên thông tin được cung cấp.

Khi trả lời:
1.  **Tóm tắt và trình bày thông tin liên quan** một cách tự nhiên và mạch lạc.
2.  **Cung cấp các liên kết (URLs)** của các nguồn đã được tham khảo ở cuối câu trả lời để người dùng có thể kiểm tra thêm.
3.  **Giải thích rõ ràng và dễ hiểu**.
4.  **Trình bày bằng tiếng Việt chuẩn xác.**

Thông tin tham khảo:
{context}

Câu hỏi: {question}

Trả lời:"""
