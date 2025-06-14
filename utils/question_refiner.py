import re
import json
from typing import Dict, List, Tuple, Optional, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from config import Config

class VietnameseLegalQuestionRefiner:
    """
    Refines Vietnamese legal questions for better search and understanding
    """
    
    def __init__(self):
        # Initialize LLM for question refinement
        self.llm = None
        if Config.GOOGLE_API_KEY:
            self.llm = ChatGoogleGenerativeAI(
                model=Config.MODEL_REFINE,
                google_api_key=Config.GOOGLE_API_KEY,
                temperature=0.1
            )
        
        # Vietnamese legal terminology mappings
        self.legal_abbreviations = {
            # Common legal abbreviations
            "dn": "doanh nghiệp",
            "dntn": "doanh nghiệp tư nhân", 
            "tnhh": "trách nhiệm hữu hạn",
            "cp": "cổ phần",
            "hđ": "hợp đồng",
            "hđlđ": "hợp đồng lao động",
            "tclđ": "tai cạnh lao động",
            "bhxh": "bảo hiểm xã hội",
            "bhyt": "bảo hiểm y tế",
            "bhtn": "bảo hiểm thất nghiệp",
            "qsd": "quyền sử dụng",
            "qsdđ": "quyền sử dụng đất",
            "gcn": "giấy chứng nhận",
            "gpkd": "giấy phép kinh doanh",
            "gpđkkd": "giấy phép đăng ký kinh doanh",
            "mst": "mã số thuế",
            "tncn": "thuế thu nhập cá nhân",
            "tndn": "thuế thu nhập doanh nghiệp",
            "gtgt": "giá trị gia tăng",
            "vat": "thuế giá trị gia tăng",
            "nld": "người lao động",
            "ntd": "người sử dụng lao động",
            "tc": "tài chính",
            "kt": "kế toán",
            "tl": "tài liệu",
            "vb": "văn bản",
            "qđ": "quyết định",
            "tt": "thông tư",
            "nđ": "nghị định",
            "dl": "dự luật",
            "qh": "quốc hội",
            "cp": "chính phủ",
            "btc": "bộ tài chính",
            "blđtbxh": "bộ lao động thương binh và xã hội",
            "btp": "bộ tư pháp",
            "btn": "bộ tài nguyên",
            "khdn": "kế hoạch doanh nghiệp"
        }
        
        # Legal context keywords
        self.legal_contexts = {
            "business": ["doanh nghiệp", "kinh doanh", "công ty", "thành lập", "giải thể", "vốn điều lệ"],
            "labor": ["lao động", "nhân viên", "hợp đồng lao động", "lương", "nghỉ phép", "sa thải"],
            "tax": ["thuế", "kê khai", "miễn thuế", "giảm thuế", "mức thuế", "thuế suất"],
            "real_estate": ["bất động sản", "đất đai", "nhà ở", "chuyển nhượng", "sổ đỏ", "quyền sử dụng"],
            "family": ["gia đình", "hôn nhân", "ly hôn", "thừa kế", "con cái", "nuôi dưỡng"],
            "criminal": ["hình sự", "vi phạm", "tội danh", "án phạt", "bồi thường"],
            "civil": ["dân sự", "tranh chấp", "khiếu nại", "tố cáo", "bồi thường"]
        }
        
        # Common misspellings and corrections
        self.common_corrections = {
            "doanh nghiep": "doanh nghiệp",
            "hop dong": "hợp đồng",
            "lao dong": "lao động",
            "tai chinh": "tài chính",
            "ke toan": "kế toán",
            "thue": "thuế",
            "quyen": "quyền",
            "nghia vu": "nghĩa vụ",
            "dat dai": "đất đai",
            "nha o": "nhà ở",
            "gia dinh": "gia đình",
            "hon nhan": "hôn nhân",
            "ly hon": "ly hôn"
        }
    
    def refine_question(self, question: str, use_llm: bool = True) -> Dict[str, str]:
        """
        Main method to refine a Vietnamese legal question
        
        Args:
            question: Original user question
            use_llm: Whether to use LLM for advanced refinement
            
        Returns:
            Dictionary containing original and refined questions with metadata
        """
        result = {
            "original_question": question,
            "refined_question": question,
            "refinement_steps": [],
            "detected_context": [],
            "expanded_terms": [],
            "corrections_made": []
        }
        
        # Step 1: Basic cleaning and normalization
        cleaned_question = self._basic_cleaning(question)
        if cleaned_question != question:
            result["refinement_steps"].append("basic_cleaning")
            result["refined_question"] = cleaned_question
        
        # Step 2: Correct common misspellings
        corrected_question = self._correct_spelling(cleaned_question)
        if corrected_question != cleaned_question:
            result["refinement_steps"].append("spelling_correction")
            result["corrections_made"] = self._get_corrections_made(cleaned_question, corrected_question)
            result["refined_question"] = corrected_question
        
        # Step 3: Expand abbreviations
        expanded_question = self._expand_abbreviations(corrected_question)
        if expanded_question != corrected_question:
            result["refinement_steps"].append("abbreviation_expansion")
            result["expanded_terms"] = self._get_expanded_terms(corrected_question, expanded_question)
            result["refined_question"] = expanded_question
        
        # Step 4: Advanced LLM-based context detection (if enabled)
        if use_llm and self.llm:
            context = self._llm_detect_legal_context(expanded_question)
            result["llm_context_detection"] = True
        else:
            context = self._detect_legal_context(expanded_question)
            result["llm_context_detection"] = False
        
        result["detected_context"] = context
        
        # Step 5: LLM-based intent analysis (if enabled)
        intent_analysis = {}
        if use_llm and self.llm:
            intent_analysis = self._llm_analyze_question_intent(expanded_question)
            result["intent_analysis"] = intent_analysis
            result["refinement_steps"].append("intent_analysis")
        
        # Step 6: Add context keywords
        context_enhanced_question = self._add_context_keywords(expanded_question, context)
        if context_enhanced_question != expanded_question:
            result["refinement_steps"].append("context_enhancement")
            result["refined_question"] = context_enhanced_question
        
        # Step 7: Advanced LLM-based refinement (if enabled and available)
        if use_llm and self.llm and len(result["refined_question"].strip()) > 10:
            best_refined = result["refined_question"]
            refinement_method = None
            
            # Try chain-of-thought for complex questions
            if (Config.ENABLE_CHAIN_OF_THOUGHT and 
                intent_analysis.get("complexity") == "complex"):
                cot_refined = self._llm_chain_of_thought_refinement(
                    result["refined_question"], context, intent_analysis
                )
                if cot_refined:
                    best_refined = cot_refined
                    refinement_method = "chain_of_thought"
                    result["refinement_steps"].append("chain_of_thought")
            
            # Try iterative refinement for moderate complexity
            elif (Config.ENABLE_ITERATIVE_REFINEMENT and 
                  intent_analysis.get("complexity") in ["moderate", "complex"]):
                iterative_refined = self._llm_iterative_refinement(
                    result["refined_question"], context, Config.MAX_REFINEMENT_ITERATIONS
                )
                if iterative_refined:
                    best_refined = iterative_refined
                    refinement_method = "iterative"
                    result["refinement_steps"].append("iterative_refinement")
            
            # Fallback to standard advanced refinement
            if not refinement_method:
                llm_refined = self._llm_refine_question_advanced(
                    result["refined_question"], 
                    context, 
                    intent_analysis
                )
                if llm_refined:
                    best_refined = llm_refined
                    refinement_method = "advanced"
                    result["refinement_steps"].append("llm_enhancement")
            
            # Apply the best refinement if found
            if best_refined != result["refined_question"]:
                # Validate the refinement with LLM (if enabled)
                if Config.ENABLE_LLM_VALIDATION:
                    if self._llm_validate_refinement(result["refined_question"], best_refined, context):
                        result["refined_question"] = best_refined
                        result["llm_validation_passed"] = True
                        result["refinement_method"] = refinement_method
                    else:
                        result["llm_validation_passed"] = False
                        print(f"LLM refinement ({refinement_method}) rejected by validation")
                else:
                    # Apply without validation
                    result["refined_question"] = best_refined
                    result["llm_validation_passed"] = None
                    result["refinement_method"] = refinement_method
        
        return result
    
    def _basic_cleaning(self, question: str) -> str:
        """Basic text cleaning and normalization"""
        # Remove extra whitespace
        question = re.sub(r'\s+', ' ', question.strip())
        
        # Remove special characters except Vietnamese diacritics and basic punctuation
        question = re.sub(r'[^\w\s\u00C0-\u017F\u1EA0-\u1EF9\?\.\,\!\-\(\)]', ' ', question)
        
        # Normalize question marks
        question = re.sub(r'\?+', '?', question)
        
        # Ensure question ends with appropriate punctuation
        if not question.endswith(('?', '.', '!')):
            question += '?'
        
        return question.strip()
    
    def _correct_spelling(self, question: str) -> str:
        """Correct common Vietnamese legal term misspellings"""
        corrected = question.lower()
        
        for misspelling, correction in self.common_corrections.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(misspelling) + r'\b'
            corrected = re.sub(pattern, correction, corrected, flags=re.IGNORECASE)
        
        return corrected
    
    def _expand_abbreviations(self, question: str) -> str:
        """Expand common Vietnamese legal abbreviations"""
        expanded = question.lower()
        
        for abbrev, full_form in self.legal_abbreviations.items():
            # Match abbreviations with word boundaries
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            # Replace with both abbreviated and full form for better search
            replacement = f"{abbrev} {full_form}"
            expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
        
        return expanded
    
    def _detect_legal_context(self, question: str) -> List[str]:
        """Detect the legal context/domain of the question"""
        detected_contexts = []
        question_lower = question.lower()
        
        for context, keywords in self.legal_contexts.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_contexts.append(context)
        
        return detected_contexts
    
    def _add_context_keywords(self, question: str, contexts: List[str]) -> str:
        """Add relevant context keywords to improve search"""
        if not contexts:
            return question
        
        # Add general legal keywords
        enhanced = question
        
        # Add context-specific keywords
        context_keywords = []
        for context in contexts:
            if context == "business":
                context_keywords.extend(["luật doanh nghiệp", "đăng ký kinh doanh"])
            elif context == "labor":
                context_keywords.extend(["bộ luật lao động", "quyền lao động"])
            elif context == "tax":
                context_keywords.extend(["luật thuế", "nghĩa vụ thuế"])
            elif context == "real_estate":
                context_keywords.extend(["luật đất đai", "quyền sở hữu"])
            elif context == "family":
                context_keywords.extend(["luật hôn nhân gia đình"])
        
        # Add keywords that aren't already in the question
        question_lower = question.lower()
        new_keywords = [kw for kw in context_keywords if kw not in question_lower]
        
        if new_keywords:
            enhanced = f"{question} {' '.join(new_keywords[:2])}"  # Add max 2 keywords
        
        return enhanced
    
    def _llm_detect_legal_context(self, question: str) -> List[str]:
        """Use LLM to detect legal context more accurately"""
        if not self.llm:
            return self._detect_legal_context(question)
        
        prompt = PromptTemplate(
            template="""Bạn là chuyên gia phân loại câu hỏi pháp luật Việt Nam. Hãy phân tích câu hỏi và xác định lĩnh vực pháp lý liên quan.

Các lĩnh vực pháp lý chính:
- business: Doanh nghiệp, kinh doanh, thành lập công ty, giải thể, vốn điều lệ
- labor: Lao động, hợp đồng lao động, sa thải, lương, nghỉ phép, bảo hiểm xã hội
- tax: Thuế, kê khai thuế, miễn thuế, thuế thu nhập, VAT
- real_estate: Bất động sản, đất đai, nhà ở, chuyển nhượng, sở hữu
- family: Hôn nhân, ly hôn, thừa kế, nuôi con, quyền con cái
- criminal: Hình sự, tội phạm, vi phạm, án phạt, truy tố
- civil: Dân sự, hợp đồng, tranh chấp, bồi thường, quyền sở hữu
- administrative: Hành chính, thủ tục, giấy tờ, cơ quan nhà nước
- constitutional: Hiến pháp, quyền công dân, nghĩa vụ, cơ cấu nhà nước

Câu hỏi: {question}

Hãy trả về tối đa 3 lĩnh vực phù hợp nhất, cách nhau bởi dấu phẩy (ví dụ: business, tax):""",
            input_variables=["question"]
        )
        
        try:
            response = self.llm.invoke(prompt.format(question=question))
            contexts = [ctx.strip() for ctx in response.content.strip().split(",")]
            # Validate contexts
            valid_contexts = ["business", "labor", "tax", "real_estate", "family", "criminal", "civil", "administrative", "constitutional"]
            return [ctx for ctx in contexts if ctx in valid_contexts][:3]
        except Exception as e:
            print(f"Error in LLM context detection: {e}")
            return self._detect_legal_context(question)
    
    def _llm_analyze_question_intent(self, question: str) -> Dict[str, Any]:
        """Use LLM to analyze question intent and structure"""
        if not self.llm:
            return self._get_fallback_intent_analysis(question)
        
        # Simplified prompt for more reliable JSON response
        prompt = PromptTemplate(
            template="""Analyze this Vietnamese legal question and return ONLY a JSON object.

Question: {question}

Return JSON with these exact fields:
- intent: "procedural" OR "definition" OR "comparison" OR "calculation" OR "advice" OR "specific_case"
- complexity: "simple" OR "moderate" OR "complex" 
- keywords: array of 3-5 Vietnamese keywords
- ambiguity_level: "low" OR "medium" OR "high"
- requires_clarification: true OR false

Example: {{"intent": "procedural", "complexity": "simple", "keywords": ["thành lập", "doanh nghiệp"], "ambiguity_level": "low", "requires_clarification": false}}

ONLY return the JSON object, no other text:""",
            input_variables=["question"]
        )
        
        try:
            response = self.llm.invoke(prompt.format(question=question))
            
            # Check if response exists and has content
            if not response or not hasattr(response, 'content'):
                print("Empty or invalid response from LLM")
                return self._get_fallback_intent_analysis(question)
            
            content = response.content.strip()
            
            # Debug: print raw response
            print(f"Raw LLM response: '{content[:100]}...'")
            
            if not content:
                print("Empty content from LLM")
                return self._get_fallback_intent_analysis(question)
            
            # Clean up the response to extract JSON
            json_content = self._extract_json_from_response(content)
            
            if not json_content:
                print("No JSON found in response")
                return self._get_fallback_intent_analysis(question)
            
            # Parse JSON
            analysis = json.loads(json_content)
            
            # Validate the analysis fields
            validated_analysis = self._validate_intent_analysis(analysis)
            print(f"Validated analysis: {validated_analysis}")
            return validated_analysis
                
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Attempted to parse: '{json_content if 'json_content' in locals() else 'N/A'}'")
            return self._get_fallback_intent_analysis(question)
        except Exception as e:
            print(f"Error in LLM intent analysis: {e}")
            # Try a simplified backup approach
            return self._simple_llm_intent_analysis(question)
    
    def _extract_json_from_response(self, content: str) -> Optional[str]:
        """Extract JSON from LLM response that might contain extra text"""
        if not content or not content.strip():
            return None
        
        content = content.strip()
        
        # Remove markdown code blocks if present
        content = re.sub(r'```json\s*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'```\s*$', '', content)
        content = re.sub(r'^```\s*', '', content)
        
        # Remove common prefixes
        prefixes_to_remove = [
            "here is the json:",
            "here's the json:",
            "json:",
            "response:",
            "analysis:",
        ]
        
        content_lower = content.lower()
        for prefix in prefixes_to_remove:
            if content_lower.startswith(prefix):
                content = content[len(prefix):].strip()
                break
        
        # If content already looks like JSON (starts with {), try to use it directly
        if content.startswith('{') and content.endswith('}'):
            return content
        
        # Try to find JSON object in the response using regex
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, content, re.DOTALL)
        
        if matches:
            # Return the first (hopefully only) JSON match
            return matches[0].strip()
        
        # If no JSON pattern found, try to extract between first { and last }
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            extracted = content[start_idx:end_idx + 1].strip()
            # Basic validation - should have at least one : for key-value pairs
            if ':' in extracted:
                return extracted
        
        return None
    
    def _simple_llm_intent_analysis(self, question: str) -> Dict[str, Any]:
        """Simplified LLM analysis with basic prompts"""
        if not self.llm:
            return self._get_fallback_intent_analysis(question)
        
        try:
            # Very simple approach - ask for specific fields one by one
            intent_prompt = f"What type of legal question is this? Answer only: procedural, definition, comparison, calculation, advice, or specific_case\n\nQuestion: {question}\n\nAnswer:"
            complexity_prompt = f"How complex is this question? Answer only: simple, moderate, or complex\n\nQuestion: {question}\n\nAnswer:"
            
            intent_response = self.llm.invoke(intent_prompt)
            complexity_response = self.llm.invoke(complexity_prompt)
            
            # Extract simple responses
            intent = intent_response.content.strip().lower()
            complexity = complexity_response.content.strip().lower()
            
            # Validate responses
            valid_intents = ["procedural", "definition", "comparison", "calculation", "advice", "specific_case"]
            valid_complexity = ["simple", "moderate", "complex"]
            
            if intent not in valid_intents:
                intent = "procedural"  # default
            if complexity not in valid_complexity:
                complexity = "simple"  # default
            
            # Extract keywords using simple approach
            keywords = []
            words = question.lower().split()
            important_words = [w for w in words if len(w) > 3 and w not in ["thế", "nào", "như", "thì", "này", "được", "có", "của", "để", "cho", "với", "trong", "từ", "về"]]
            keywords = important_words[:3]
            
            return {
                "intent": intent,
                "complexity": complexity,
                "keywords": keywords,
                "ambiguity_level": "medium" if complexity == "complex" else "low",
                "requires_clarification": complexity == "complex"
            }
            
        except Exception as e:
            print(f"Error in simple LLM analysis: {e}")
            return self._get_fallback_intent_analysis(question)
    
    def _validate_intent_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean up intent analysis results"""
        validated = {}
        
        # Validate intent
        valid_intents = ["procedural", "definition", "comparison", "calculation", "advice", "specific_case"]
        validated["intent"] = analysis.get("intent", "unknown")
        if validated["intent"] not in valid_intents:
            validated["intent"] = "unknown"
        
        # Validate complexity
        valid_complexity = ["simple", "moderate", "complex"]
        validated["complexity"] = analysis.get("complexity", "simple")
        if validated["complexity"] not in valid_complexity:
            validated["complexity"] = "simple"
        
        # Validate keywords
        keywords = analysis.get("keywords", [])
        if isinstance(keywords, list):
            validated["keywords"] = [str(k).strip() for k in keywords[:5] if k and str(k).strip()]
        else:
            validated["keywords"] = []
        
        # Validate ambiguity level
        valid_ambiguity = ["low", "medium", "high"]
        validated["ambiguity_level"] = analysis.get("ambiguity_level", "low")
        if validated["ambiguity_level"] not in valid_ambiguity:
            validated["ambiguity_level"] = "low"
        
        # Validate requires_clarification
        validated["requires_clarification"] = bool(analysis.get("requires_clarification", False))
        
        # Validate suggested_clarifications
        clarifications = analysis.get("suggested_clarifications", [])
        if isinstance(clarifications, list):
            validated["suggested_clarifications"] = [str(c).strip() for c in clarifications[:3] if c and str(c).strip()]
        else:
            validated["suggested_clarifications"] = []
        
        return validated
    
    def _get_fallback_intent_analysis(self, question: str) -> Dict[str, Any]:
        """Get fallback intent analysis using rule-based approach"""
        # Simple rule-based fallback
        question_lower = question.lower()
        
        # Determine intent based on keywords
        if any(word in question_lower for word in ["thủ tục", "cách", "làm thế nào", "quy trình", "bước"]):
            intent = "procedural"
        elif any(word in question_lower for word in ["là gì", "định nghĩa", "khái niệm", "nghĩa là"]):
            intent = "definition"
        elif any(word in question_lower for word in ["so sánh", "khác nhau", "giống", "khác biệt"]):
            intent = "comparison"
        elif any(word in question_lower for word in ["tính", "tính toán", "phí", "lệ phí", "thuế"]):
            intent = "calculation"
        elif any(word in question_lower for word in ["nên", "có thể", "được không", "có được"]):
            intent = "advice"
        else:
            intent = "specific_case"
        
        # Determine complexity based on length and question marks
        word_count = len(question.split())
        if word_count < 8:
            complexity = "simple"
        elif word_count < 20:
            complexity = "moderate"
        else:
            complexity = "complex"
        
        # Extract simple keywords
        keywords = []
        for word in question.split():
            word_clean = re.sub(r'[^\w]', '', word).lower()
            if len(word_clean) > 3 and word_clean not in ["thế", "nào", "như", "thì", "này", "được", "có", "của"]:
                keywords.append(word_clean)
                if len(keywords) >= 3:
                    break
        
        return {
            "intent": intent,
            "complexity": complexity,
            "keywords": keywords,
            "ambiguity_level": "medium" if complexity == "complex" else "low",
            "requires_clarification": complexity == "complex",
            "suggested_clarifications": []
        }
    
    def _llm_refine_question_advanced(self, question: str, contexts: List[str], intent_analysis: Dict[str, Any]) -> Optional[str]:
        """Advanced LLM-based question refinement with context and intent awareness"""
        if not self.llm:
            return None
        
        context_str = ", ".join(contexts) if contexts else "tổng quát"
        intent = intent_analysis.get("intent", "unknown")
        complexity = intent_analysis.get("complexity", "simple")
        keywords = intent_analysis.get("keywords", [])
        
        # Choose refinement strategy based on intent
        if intent == "procedural":
            strategy_prompt = """
Đây là câu hỏi về thủ tục pháp lý. Hãy:
- Làm rõ loại thủ tục cụ thể
- Thêm từ khóa về quy trình, bước thực hiện
- Đề cập đến cơ quan có thẩm quyền nếu phù hợp"""
        elif intent == "definition":
            strategy_prompt = """
Đây là câu hỏi định nghĩa khái niệm. Hãy:
- Làm rõ khái niệm cần định nghĩa
- Thêm ngữ cảnh pháp lý liên quan
- Đề cập đến văn bản luật có liên quan"""
        elif intent == "comparison":
            strategy_prompt = """
Đây là câu hỏi so sánh. Hãy:
- Làm rõ các đối tượng được so sánh
- Thêm tiêu chí so sánh cụ thể
- Đảm bảo tính khách quan"""
        else:
            strategy_prompt = """
Hãy cải thiện câu hỏi theo nguyên tắc chung:
- Làm rõ ý định của câu hỏi
- Thêm ngữ cảnh pháp lý phù hợp
- Sử dụng thuật ngữ chuẩn mực"""
        
        prompt = PromptTemplate(
            template="""Bạn là chuyên gia pháp lý Việt Nam có 20 năm kinh nghiệm. Hãy cải thiện câu hỏi pháp lý sau để tối ưu hóa việc tìm kiếm thông tin.

THÔNG TIN PHÂN TÍCH:
- Lĩnh vực pháp lý: {context}
- Loại câu hỏi: {intent}
- Độ phức tạp: {complexity}
- Từ khóa chính: {keywords}

CHIẾN LƯỢC CẢI THIỆN:
{strategy}

NGUYÊN TẮC CHUNG:
1. Giữ nguyên ý nghĩa gốc của câu hỏi
2. Sử dụng thuật ngữ pháp lý chính xác và chuẩn mực
3. Làm rõ các khái niệm mơ hồ
4. Thêm ngữ cảnh pháp lý cần thiết
5. Tối ưu hóa cho tìm kiếm trong cơ sở dữ liệu pháp luật
6. Đảm bảo câu hỏi ngắn gọn nhưng đầy đủ thông tin
7. Ưu tiên các từ khóa xuất hiện trong văn bản pháp luật Việt Nam

Câu hỏi gốc: {question}

Câu hỏi được cải thiện (chỉ trả về câu hỏi, không giải thích):""",
            input_variables=["question", "context", "intent", "complexity", "keywords", "strategy"]
        )
        
        try:
            response = self.llm.invoke(prompt.format(
                question=question,
                context=context_str,
                intent=intent,
                complexity=complexity,
                keywords=", ".join(keywords),
                strategy=strategy_prompt
            ))
            
            refined = response.content.strip()
            
            # Advanced validation
            if self._validate_refined_question(question, refined, intent_analysis):
                return refined
                
        except Exception as e:
            print(f"Error in advanced LLM refinement: {e}")
        
        return None
    
    def _llm_validate_refinement(self, original: str, refined: str, contexts: List[str]) -> bool:
        """Use LLM to validate if the refinement maintains original intent"""
        if not self.llm:
            return True
        
        prompt = PromptTemplate(
            template="""Bạn là chuyên gia đánh giá chất lượng câu hỏi pháp lý. Hãy đánh giá xem câu hỏi đã được cải thiện có giữ nguyên ý nghĩa gốc và có tốt hơn cho việc tìm kiếm thông tin pháp luật không.

Câu hỏi gốc: {original}
Câu hỏi đã cải thiện: {refined}
Lĩnh vực: {contexts}

Tiêu chí đánh giá:
1. Giữ nguyên ý nghĩa gốc (có/không)
2. Cải thiện khả năng tìm kiếm (có/không)
3. Sử dụng thuật ngữ pháp lý phù hợp (có/không)
4. Độ dài hợp lý (có/không)
5. Rõ ràng và dễ hiểu (có/không)

Kết luận: CHẤP_NHẬN hoặc TỪ_CHỐI

Chỉ trả về kết luận:""",
            input_variables=["original", "refined", "contexts"]
        )
        
        try:
            response = self.llm.invoke(prompt.format(
                original=original,
                refined=refined,
                contexts=", ".join(contexts)
            ))
            
            return "CHẤP_NHẬN" in response.content.strip().upper()
        except Exception as e:
            print(f"Error in LLM validation: {e}")
            return True
    
    def _validate_refined_question(self, original: str, refined: str, intent_analysis: Dict[str, Any]) -> bool:
        """Validate refined question with multiple criteria"""
        if not refined or not refined.strip():
            return False
        
        # Basic length check
        if len(refined) < 10 or len(refined) > 500:
            return False
        
        # Should contain question mark for questions
        if intent_analysis.get("intent") in ["procedural", "definition"] and "?" not in refined:
            return False
        
        # Shouldn't start with meta phrases
        meta_phrases = ["câu hỏi", "tôi muốn hỏi", "xin hỏi", "cho tôi biết"]
        if any(refined.lower().startswith(phrase) for phrase in meta_phrases):
            return False
        
        # Should be different from original (some improvement made)
        if refined.strip().lower() == original.strip().lower():
            return False
        
        return True
    
    def _get_corrections_made(self, original: str, corrected: str) -> List[Dict[str, str]]:
        """Get list of spelling corrections that were made"""
        corrections = []
        for misspelling, correction in self.common_corrections.items():
            if misspelling in original.lower() and correction in corrected.lower():
                corrections.append({"from": misspelling, "to": correction})
        return corrections
    
    def _get_expanded_terms(self, original: str, expanded: str) -> List[Dict[str, str]]:
        """Get list of abbreviations that were expanded"""
        expansions = []
        for abbrev, full_form in self.legal_abbreviations.items():
            if abbrev in original.lower() and full_form in expanded.lower():
                expansions.append({"abbreviation": abbrev, "full_form": full_form})
        return expansions
    
    def get_refinement_summary(self, refinement_result: Dict) -> str:
        """Generate a human-readable summary of refinements made"""
        if not refinement_result["refinement_steps"]:
            return "Không có cải thiện nào được thực hiện."
        
        summary_parts = []
        
        if "basic_cleaning" in refinement_result["refinement_steps"]:
            summary_parts.append("làm sạch văn bản")
        
        if "spelling_correction" in refinement_result["refinement_steps"]:
            corrections = refinement_result["corrections_made"]
            if corrections:
                summary_parts.append(f"sửa {len(corrections)} lỗi chính tả")
        
        if "abbreviation_expansion" in refinement_result["refinement_steps"]:
            expansions = refinement_result["expanded_terms"]
            if expansions:
                summary_parts.append(f"mở rộng {len(expansions)} từ viết tắt")
        
        if "intent_analysis" in refinement_result["refinement_steps"]:
            intent_analysis = refinement_result.get("intent_analysis", {})
            intent = intent_analysis.get("intent", "unknown")
            complexity = intent_analysis.get("complexity", "simple")
            summary_parts.append(f"phân tích ý định ({intent}, độ phức tạp: {complexity})")
        
        if "context_enhancement" in refinement_result["refinement_steps"]:
            contexts = refinement_result["detected_context"]
            if contexts:
                context_method = "AI" if refinement_result.get("llm_context_detection") else "quy tắc"
                summary_parts.append(f"thêm từ khóa cho lĩnh vực {', '.join(contexts)} ({context_method})")
        
        # LLM enhancements
        llm_methods = []
        if "chain_of_thought" in refinement_result["refinement_steps"]:
            llm_methods.append("suy luận từng bước")
        if "iterative_refinement" in refinement_result["refinement_steps"]:
            llm_methods.append("cải thiện lặp")
        if "llm_enhancement" in refinement_result["refinement_steps"]:
            llm_methods.append("cải thiện tiêu chuẩn")
        
        if llm_methods:
            validation_status = ""
            if refinement_result.get("llm_validation_passed") is not None:
                validation_status = " (đã xác thực)" if refinement_result["llm_validation_passed"] else " (chưa xác thực)"
            
            method_str = ", ".join(llm_methods)
            summary_parts.append(f"cải thiện bằng AI ({method_str}){validation_status}")
        
        return f"Đã {', '.join(summary_parts)}."
    
    def get_detailed_analysis(self, refinement_result: Dict) -> str:
        """Get detailed analysis of the refinement process"""
        if not refinement_result.get("intent_analysis"):
            return ""
        
        intent_analysis = refinement_result["intent_analysis"]
        analysis_parts = []
        
        # Intent information
        intent = intent_analysis.get("intent", "unknown")
        intent_map = {
            "procedural": "Thủ tục",
            "definition": "Định nghĩa", 
            "comparison": "So sánh",
            "calculation": "Tính toán",
            "advice": "Tư vấn",
            "specific_case": "Trường hợp cụ thể"
        }
        analysis_parts.append(f"Loại câu hỏi: {intent_map.get(intent, intent)}")
        
        # Complexity
        complexity = intent_analysis.get("complexity", "simple")
        complexity_map = {"simple": "Đơn giản", "moderate": "Trung bình", "complex": "Phức tạp"}
        analysis_parts.append(f"Độ phức tạp: {complexity_map.get(complexity, complexity)}")
        
        # Keywords
        keywords = intent_analysis.get("keywords", [])
        if keywords:
            analysis_parts.append(f"Từ khóa chính: {', '.join(keywords[:3])}")
        
        # Ambiguity level
        ambiguity = intent_analysis.get("ambiguity_level", "low")
        ambiguity_map = {"low": "Thấp", "medium": "Trung bình", "high": "Cao"}
        analysis_parts.append(f"Độ mơ hồ: {ambiguity_map.get(ambiguity, ambiguity)}")
        
        return " | ".join(analysis_parts)
    
    def _llm_chain_of_thought_refinement(self, question: str, contexts: List[str], intent_analysis: Dict[str, Any]) -> Optional[str]:
        """Use chain-of-thought reasoning for complex question refinement"""
        if not self.llm or intent_analysis.get("complexity") != "complex":
            return None
        
        prompt = PromptTemplate(
            template="""Bạn là chuyên gia pháp lý Việt Nam với 25 năm kinh nghiệm. Hãy sử dụng phương pháp suy luận từng bước để cải thiện câu hỏi pháp lý phức tạp sau.

THÔNG TIN PHÂN TÍCH:
- Câu hỏi gốc: {question}
- Lĩnh vực pháp lý: {contexts}
- Độ phức tạp: {complexity}
- Độ mơ hồ: {ambiguity}

BƯỚC 1: PHÂN TÍCH VẤN ĐỀ
Hãy xác định:
- Vấn đề pháp lý cốt lõi là gì?
- Có những khái niệm nào cần làm rõ?
- Thiếu thông tin gì để trả lời đầy đủ?

BƯỚC 2: XÁC ĐỊNH NGỮ CẢNH PHÁP LÝ
Hãy xác định:
- Văn bản pháp luật nào có khả năng liên quan?
- Cơ quan có thẩm quyền nào cần đề cập?
- Thủ tục hoặc quy trình nào cần nêu rõ?

BƯỚC 3: TỐI ƯU HÓA TỪ KHÓA
Hãy xác định:
- Thuật ngữ pháp lý chính xác cần sử dụng
- Từ khóa tìm kiếm hiệu quả
- Cụm từ thường xuất hiện trong văn bản pháp luật

BƯỚC 4: XÂY DỰNG CÂU HỎI TỐI ƯU
Dựa trên 3 bước trên, hãy xây dựng câu hỏi mới:
- Rõ ràng và cụ thể
- Sử dụng thuật ngữ pháp lý chuẩn
- Tối ưu cho tìm kiếm

ĐỊNH DẠNG TRẢ LỜI JSON:
{{
    "analysis": {{
        "core_legal_issue": "vấn đề pháp lý cốt lõi",
        "unclear_concepts": ["khái niệm 1", "khái niệm 2"],
        "missing_information": ["thông tin thiếu 1", "thông tin thiếu 2"]
    }},
    "legal_context": {{
        "relevant_laws": ["luật 1", "luật 2"],
        "authorities": ["cơ quan 1", "cơ quan 2"],
        "procedures": ["thủ tục 1", "thủ tục 2"]
    }},
    "keywords": {{
        "legal_terms": ["thuật ngữ 1", "thuật ngữ 2"],
        "search_keywords": ["từ khóa 1", "từ khóa 2"],
        "legal_phrases": ["cụm từ 1", "cụm từ 2"]
    }},
    "refined_question": "câu hỏi được cải thiện",
    "confidence_score": 0.95,
    "reasoning": "lý do tại sao câu hỏi này tốt hơn"
}}

Chỉ trả về JSON hợp lệ:""",
            input_variables=["question", "contexts", "complexity", "ambiguity"]
        )
        
        try:
            response = self.llm.invoke(prompt.format(
                question=question,
                contexts=", ".join(contexts),
                complexity=intent_analysis.get("complexity", "complex"),
                ambiguity=intent_analysis.get("ambiguity_level", "high")
            ))
            
            result = json.loads(response.content.strip())
            refined_question = result.get("refined_question", "")
            confidence = result.get("confidence_score", 0.0)
            
            # Only return if confidence is high enough
            if refined_question and confidence > Config.MIN_CONFIDENCE_SCORE:
                return refined_question
                
        except Exception as e:
            print(f"Error in chain-of-thought refinement: {e}")
        
        return None
    
    def _llm_iterative_refinement(self, question: str, contexts: List[str], max_iterations: int = 3) -> Optional[str]:
        """Use iterative refinement to progressively improve the question"""
        if not self.llm:
            return None
        
        current_question = question
        
        for iteration in range(max_iterations):
            prompt = PromptTemplate(
                template="""Bạn là chuyên gia cải thiện câu hỏi pháp lý. Đây là lần cải thiện thứ {iteration} của câu hỏi.

Câu hỏi hiện tại: {current_question}
Lĩnh vực pháp lý: {contexts}

Hãy phân tích và cải thiện thêm câu hỏi theo các tiêu chí:

LẦN 1: Tập trung vào thuật ngữ pháp lý và cấu trúc câu
LẦN 2: Tập trung vào ngữ cảnh và từ khóa tìm kiếm
LẦN 3: Tập trung vào tính rõ ràng và độ chính xác

Nguyên tắc cải thiện:
1. Mỗi lần cải thiện phải có tiến bộ rõ rệt
2. Giữ nguyên ý nghĩa gốc
3. Tăng cường khả năng tìm kiếm
4. Sử dụng thuật ngữ chuẩn mực

Trả về định dạng JSON:
{{
    "improved_question": "câu hỏi được cải thiện",
    "improvements_made": ["cải thiện 1", "cải thiện 2"],
    "quality_score": 0.85,
    "needs_further_improvement": true/false
}}

Chỉ trả về JSON:""",
                input_variables=["current_question", "contexts", "iteration"]
            )
            
            try:
                response = self.llm.invoke(prompt.format(
                    current_question=current_question,
                    contexts=", ".join(contexts),
                    iteration=iteration + 1
                ))
                
                result = json.loads(response.content.strip())
                improved_question = result.get("improved_question", "")
                quality_score = result.get("quality_score", 0.0)
                needs_improvement = result.get("needs_further_improvement", False)
                
                if improved_question and improved_question != current_question:
                    current_question = improved_question
                    
                    # Stop if quality is high enough or no further improvement needed
                    if quality_score > 0.9 or not needs_improvement:
                        break
                else:
                    break
                    
            except Exception as e:
                print(f"Error in iterative refinement iteration {iteration + 1}: {e}")
                break
        
        return current_question if current_question != question else None 