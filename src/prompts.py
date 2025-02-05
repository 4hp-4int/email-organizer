AGENT_SCRAPER_PROMPT = """
You are an expert in HTML and DOM analysis.
Below is a raw representation of the page's DOM structure:
DOM Map: {json.dumps(dom_representation['dom_map'], indent=2)}
Snippets (first 300 chars of each chunk):
{json.dumps(dom_representation['snippets'], indent=2)}

Now, answer the following question about the DOM:
{question}

Return your answer in JSON format.
"""
