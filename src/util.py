from typing import List, Dict

from bs4 import BeautifulSoup


def chunk_text(text: str, max_length: int) -> List[str]:
    return [text[i : i + max_length] for i in range(0, len(text), max_length)]


def process_chunk(chunk: str) -> Dict:
    soup = BeautifulSoup(chunk, "html.parser")
    # Example: Count certain tags as a crude DOM "map"
    dom_map = {
        "div": len(soup.find_all("div")),
        "p": len(soup.find_all("p")),
        "a": len(soup.find_all("a")),
        "h1": len(soup.find_all("h1")),
        "h2": len(soup.find_all("h2")),
    }
    # Also return a snippet of raw HTML for context
    snippet = chunk[:300]  # first 300 characters
    return {"dom_map": dom_map, "snippet": snippet}


def simplify_html(html_content: str):

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style", "img"]):
        tag.decompose()

    # Optionally, you might also remove comments and extra whitespace
    clean_html = soup.prettify()
    return clean_html
