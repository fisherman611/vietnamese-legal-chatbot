import requests
from googlesearch import search
from bs4 import BeautifulSoup
from typing import List, Dict
import time
from config import Config
from urllib.parse import urlparse


class GoogleSearchTool:
    """Google Search tool for legal questions with insufficient information"""

    def __init__(self):
        self.search_delay = 1

    def search_legal_info(
        self, query: str, num_results: int = None
    ) -> List[Dict[str, str]]:
        if num_results is None:
            num_results = Config.GOOGLE_SEARCH_RESULTS_COUNT

        try:
            # Enhanced Vietnamese legal query patterns
            enhanced_queries = [
                f"{query} luật pháp Việt Nam site:thuvienphapluat.vn",
                f"{query} luật pháp Việt Nam site: luatvietnam.vn"
                f"{query} pháp luật Việt Nam site:moj.gov.vn",
                f"{query} quy định pháp luật Việt Nam",
                f"{query} luật việt nam điều khoản",
            ]

            all_results = []
            seen_urls = set()

            # Try different search queries to get better results
            for enhanced_query in enhanced_queries:
                if len(all_results) >= num_results:
                    break

                try:
                    search_results = search(enhanced_query, num_results=3, lang="vi")

                    for url in search_results:
                        if len(all_results) >= num_results:
                            break

                        if url in seen_urls:
                            continue

                        seen_urls.add(url)

                        try:
                            # Get page content
                            content = self._get_page_content(url)
                            if content and content.get("snippet"):
                                all_results.append(
                                    {
                                        "url": url,
                                        "title": content.get(
                                            "title", "Không có tiêu đề"
                                        ),
                                        "snippet": content.get(
                                            "snippet", "Không có nội dung"
                                        ),
                                        "domain": self._extract_domain(url),
                                    }
                                )

                            time.sleep(self.search_delay)

                        except Exception as e:
                            print(f"Error fetching content from {url}: {e}")
                            continue

                except Exception as e:
                    print(f"Error with search query '{enhanced_query}': {e}")
                    continue

            return all_results[:num_results]

        except Exception as e:
            print(f"Error performing Google search: {e}")
            return []

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return "Unknown"

    def _get_page_content(self, url: str) -> Dict[str, str]:
        """Extract content from a web page with better Vietnamese content handling"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }

            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            # Handle encoding for Vietnamese content
            if response.encoding.lower() in ["iso-8859-1", "windows-1252"]:
                response.encoding = "utf-8"

            soup = BeautifulSoup(response.content, "html.parser")

            # Extract title
            title_tag = soup.find("title")
            title = title_tag.get_text().strip() if title_tag else "Không có tiêu đề"

            # Remove unwanted elements
            for element in soup(
                ["script", "style", "nav", "header", "footer", "aside", "iframe"]
            ):
                element.decompose()

            # Try to find main content areas
            main_content = None
            content_selectors = [
                "article",
                "main",
                ".content",
                ".post-content",
                ".entry-content",
                ".article-content",
                ".news-content",
                "#content",
                ".main-content",
            ]

            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break

            # If no main content found, use body
            if not main_content:
                main_content = soup.find("body")

            if main_content:
                text = main_content.get_text()
            else:
                text = soup.get_text()

            # Clean up text for Vietnamese content
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk and len(chunk) > 3)

            # Extract meaningful snippet (prioritize Vietnamese legal terms)
            legal_keywords = [
                "luật",
                "điều",
                "khoản",
                "quy định",
                "nghị định",
                "thông tư",
                "quyền",
                "nghĩa vụ",
            ]

            # Try to find sentences with legal keywords
            sentences = text.split(".")
            relevant_sentences = []

            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in legal_keywords):
                    relevant_sentences.append(sentence.strip())
                    if len(" ".join(relevant_sentences)) > 400:
                        break

            if relevant_sentences:
                snippet = ". ".join(relevant_sentences[:3])
            else:
                snippet = text[:600] + "..." if len(text) > 600 else text

            return {"title": title, "snippet": snippet}

        except Exception as e:
            print(f"Error extracting content from {url}: {e}")
            return {}

    def format_search_results(self, results: List[Dict[str, str]]) -> str:
        """Format search results for LLM context"""
        if not results:
            return "Không tìm thấy thông tin liên quan."

        formatted_results = ""

        for i, result in enumerate(results, 1):
            formatted_results += f"**Nguồn {i}: {result['title']}**\n"
            formatted_results += f"Website: {result.get('domain', 'Unknown')}\n"
            formatted_results += f"Nội dung: {result['snippet']}\n"
            formatted_results += f"Link: {result['url']}\n\n"

        return formatted_results

    def format_search_results_for_display(self, results: List[Dict[str, str]]) -> str:
        """Format search results for UI display with clickable links"""
        if not results:
            return "Không tìm thấy thông tin tham khảo từ web."

        # Clean HTML formatting without leading whitespaces
        formatted_html = '<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;">'
        formatted_html += '<h4 style="color: #1e40af; margin-bottom: 15px;">🌐 Nguồn tham khảo từ web:</h4>'

        for i, result in enumerate(results, 1):
            # Escape HTML characters in content
            title_escaped = result["title"].replace("<", "&lt;").replace(">", "&gt;")
            snippet_escaped = (
                result["snippet"][:200].replace("<", "&lt;").replace(">", "&gt;")
            )
            if len(result["snippet"]) > 200:
                snippet_escaped += "..."

            formatted_html += f"""<div style="background-color: white; padding: 12px; margin-bottom: 10px; border-radius: 6px; border-left: 4px solid #3b82f6;">
<h5 style="margin: 0; color: #1e40af;">
<a href="{result['url']}" target="_blank" style="text-decoration: none; color: #1e40af;">
{i}. {title_escaped}
</a>
</h5>
<p style="color: #6b7280; font-size: 0.9em; margin: 5px 0;">
📄 {result.get('domain', 'Unknown')}
</p>
<p style="margin: 8px 0; color: #374151; line-height: 1.5;">
{snippet_escaped}
</p>
<a href="{result['url']}" target="_blank" style="color: #3b82f6; text-decoration: none; font-size: 0.9em;">
🔗 Xem chi tiết →
</a>
</div>"""

        formatted_html += "</div>"
        return formatted_html
