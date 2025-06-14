import re
import pandas as pd
from typing import List, Set
from underthesea import word_tokenize
from config import Config


class VietnameseTextProcessor:
    """Vietnamese text processing utilities for legal documents"""

    def __init__(self):
        self.stopwords = self._load_stopwords()

    def _load_stopwords(self) -> Set[str]:
        """Load Vietnamese stopwords from file"""
        try:
            # Try UTF-8 first
            with open(Config.STOPWORDS_PATH, "r", encoding="utf-8") as f:
                stopwords = set(line.strip() for line in f if line.strip())
                stopwords = set(['_'.join(word.split()) for word in list(stopwords)])
            return stopwords
        except UnicodeDecodeError:
            try:
                # Try UTF-16 if UTF-8 fails
                with open(Config.STOPWORDS_PATH, "r", encoding="utf-16") as f:
                    stopwords = set(line.strip() for line in f if line.strip())
                return stopwords
            except UnicodeDecodeError:
                try:
                    # Try with BOM detection
                    with open(Config.STOPWORDS_PATH, "r", encoding="utf-8-sig") as f:
                        stopwords = set(line.strip() for line in f if line.strip())
                    return stopwords
                except UnicodeDecodeError:
                    print(
                        f"Warning: Unable to decode stopwords file at {Config.STOPWORDS_PATH}"
                    )
                    return set()
        except FileNotFoundError:
            print(f"Warning: Stopwords file not found at {Config.STOPWORDS_PATH}")
            return set()
        except Exception as e:
            print(f"Warning: Error loading stopwords file: {e}")
            return set()

    def clean_text(self, text: str) -> str:
        """Clean Vietnamese text for processing"""
        if not text:
            return ""

        # Remove extra whitespace and normalize
        text = re.sub(r"\s+", " ", text.strip())

        # Remove special characters but keep Vietnamese characters
        text = re.sub(
            r"[^\w\s\-\.\,\;\:\!\?\(\)\[\]\"\'àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđĐ]",
            " ",
            text,
        )

        # Remove multiple spaces
        text = re.sub(r"\s+", " ", text.strip())

        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize Vietnamese text using underthesea"""
        try:
            cleaned_text = self.clean_text(text)
            tokens = word_tokenize(cleaned_text, format="text").split()
            return tokens
        except Exception as e:
            print(f"Error tokenizing text: {e}")
            return text.split()

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list"""
        return [token for token in tokens if token.lower() not in self.stopwords]

    def preprocess_for_search(self, text: str) -> str:
        """Preprocess text for search - tokenize and remove stopwords"""
        tokens = self.tokenize(text)
        filtered_tokens = self.remove_stopwords(tokens)
        return " ".join(filtered_tokens)

    def extract_keywords(self, text: str, min_length: int = 2) -> List[str]:
        """Extract keywords from text"""
        tokens = self.tokenize(text)
        filtered_tokens = self.remove_stopwords(tokens)
        keywords = [token for token in filtered_tokens if len(token) >= min_length]
        return list(set(keywords))  # Remove duplicates

    def chunk_text(
        self, text: str, chunk_size: int = None, overlap: int = None
    ) -> List[str]:
        """Split text into chunks with overlap"""
        if chunk_size is None:
            chunk_size = Config.CHUNK_SIZE
        if overlap is None:
            overlap = Config.CHUNK_OVERLAP

        tokens = self.tokenize(text)
        chunks = []

        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i : i + chunk_size]
            if chunk_tokens:
                chunks.append(" ".join(chunk_tokens))

        return chunks
