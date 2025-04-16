"""
title: EuropePMC Fetcher
author: jfouret
description: Fetches scientific articles from EuropePMC API with Mistral AI query generation and Cohere reranking
version: 0.2.0
license: MIT
requirements: requests, asyncio, mistralai, cohere, pandas
"""

import requests
import asyncio
import json
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Callable, Any, Tuple

class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def emit(self, description="Unknown State", status="in_progress", done=False):
        """Emit a status update."""
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )
    
    async def emit_citation(self, article):
        """Emit a citation."""
        pmid = article.get("pmid", "")
        title = article.get("title", "No title available")
        abstract = article.get(
            "abstractText", "No abstract available"
        )
        content = f"# {title}\n\n{abstract}"

        pmid = article.get("pmid", "")

        metadata = {
            "pmid": pmid,
            "doi": article.get("doi", ""),
            "authors": article.get("authorString", ""),
            "journal": article.get("journalTitle", ""),
            "relevance": article.get("relevance", None),
            "publication_date": article.get(
                "firstPublicationDate", ""
            ),
            "source": f"EuropePMC:{pmid}",
        }
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "citation",
                    "data": {
                        "document": [content],
                        "metadata": [metadata],
                        "source": {
                            "id": pmid,
                            "name": title,
                            "url": f"https://europepmc.org/article/MED/{pmid}",
                        },
                    },
                }
            )

    async def emit_message(self, content):
        """Emit a message with the given content."""
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "message",
                    "data": {
                        "content": content
                    },
                }
            )


class Tools:
    class Valves(BaseModel):
        mistral_api_key: str = Field(
            default="",
            description="Mistral AI API key for query generation (admin only)",
        )
        cohere_api_key: str = Field(
            default="",
            description="Cohere API key for reranking (required if use_reranking is True)"
        )
        pass

    class UserValves(BaseModel):
        num_queries: int = Field(
            default=3, description="Number of queries to generate with Mistral AI (1-5)"
        )
        max_retries: int = Field(
            default=3, description="Number of retries for Mistral AI query generation if format is incorrect"
        )
        max_results: int = Field(
            default=10, description="Maximum number of results to return per query"
        )
        result_type: str = Field(
            default="core", description="Result type (core, lite, oa, etc.)"
        )
        citation_enabled: bool = Field(
            default=True, description="If True, send custom citations with links"
        )
        system_prompt: str = Field(
            default="""You are a specialized scientific literature search assistant for EuropePMC. Your task is to transform user queries into optimized search strings that leverage EuropePMC's advanced search syntax.

SEARCH SYNTAX RULES:

1. EXACT MATCH SEARCHES:
   - Use quotation marks for exact phrases: "Small cell lung cancer"
   - Without quotes, terms are connected with AND: small cell lung cancer = small AND cell AND lung AND cancer

2. BOOLEAN OPERATORS:
   - AND: Default operator, finds documents containing all terms
   - OR: Finds documents containing any of the terms
   - NOT: Excludes documents containing specific terms
   - Examples:
     * (elderly NOT children) - includes elderly, excludes children
     * (cats AND dogs) - both terms must appear
     * (cats OR dogs) - either term must appear
     * (cats NOT dogs) - cats must appear, dogs must not

3. FIELD-SPECIFIC SEARCHES:
   - TITLE:"term" - Searches only in title
   - AUTH:"Name" - Searches for specific author
   - PUB_YEAR:YYYY - Filters by publication year
   - Examples:
     * TITLE:"mouse" AND AUTH:"Chen XJ" AND PUB_YEAR:2004
     * TITLE:"mouse" NOT PUB_YEAR:2004

4. WILDCARD OPERATORS:
   - Use asterisk (*) for prefix searching
   - Example: neuron* finds neuron, neurone, neuronal, etc.

5. QUERY STRUCTURE:
   - Keep queries synthetic (concise and well-structured)
   - Prioritize specificity over breadth when appropriate
   - Use parentheses to group logical operations: (term1 OR term2) AND term3

Your goal is to generate queries that maximize relevant results while minimizing noise.""",
            description="System prompt for Mistral AI to guide query generation"
        )
        user_prompt: str = Field(
            default="""## Task
Generate optimized search queries for EuropePMC based on the user's medical or scientific research question. Your queries should cover different aspects and synonyms of the topic to ensure comprehensive search results.

Apply EuropePMC's search syntax rules:
- Use quotation marks for exact phrases: "term"
- Use Boolean operators (AND, OR, NOT) with parentheses for clarity
- Use field-specific searches when appropriate: TITLE:"term", AUTH:"Name", PUB_YEAR:YYYY
- Use wildcards (*) for prefix searching: neuron*
- Keep queries synthetic (concise and well-structured)

Generate exactly {num_queries} queries. Each query should be a string optimized for scientific literature search. Do not include any explanations or additional text outside of the JSON object.

## Initial User Query 
{user_query}

## Output format
Return ONLY a JSON object with the following format:
```json
{{
  "queries": ["query1", "query2", "query3", ...]
}}
```

## Examples
- User: "I want to know about the relationship between gut microbiome and Parkinson's disease"
- Assistant: {{"queries":[
  "\"gut microbiome\" AND \"Parkinson's disease\"",
  "intestinal microbiota AND neurodegeneration AND Parkinson*",
  "(microbiome OR microbiota) AND \"Parkinson's disease\" AND (pathophysiology OR mechanism*)",
  "TITLE:\"gut microbiome\" AND TITLE:\"Parkinson*\""
]}}

- User: "Recent advances in CRISPR gene editing for cystic fibrosis"
- Assistant: {{"queries":[
  "CRISPR AND \"cystic fibrosis\" AND (therapy OR treatment) AND PUB_YEAR:[2020 TO 2025]",
  "\"gene editing\" AND \"cystic fibrosis\" AND CFTR",
  "TITLE:CRISPR AND TITLE:\"cystic fibrosis\"",
  "(\"gene therapy\" OR \"gene editing\") AND \"cystic fibrosis\" NOT review[publication type]"
]}}""",
            description="User prompt template for Mistral AI query generation"
        )
        # New parameters for Cohere reranking
        use_reranking: bool = Field(
            default=True,
            description="Enable/disable reranking of results using Cohere"
        )
        chunk_size: int = Field(
            default=200,
            description="Size of text chunks in words for reranking"
        )
        chunk_overlap: int = Field(
            default=50,
            description="Overlap between chunks in words"
        )
        top_n: int = Field(
            default=3,
            description="Number of top results to keep after reranking"
        )
        num_queries_rerank: int = Field(
            default=1, 
            description="Number of reranking queries to generate with Mistral AI (1-3)"
        )
        rerank_query_prompt: str = Field(
            default="""## Task
Generate exactly {num_queries_rerank} queries. Each query should be human-readable ranking query based on the user's medical or scientific research question.
This queries will be used for reranking scientific articles, so it should be conversational and descriptive rather than using specialized search syntax.

Your queries should:
- Be phrased as a natural language question or statement
- Include key concepts, terms, and relationships from the original query
- Be specific enough to identify relevant content
- Be general enough to capture different phrasings of the same concepts

## Initial User Query 
{user_query}

## Output format
Return ONLY a JSON object with the following format:
```json
{{
  "queries": [
    "your natural language query here",
    "another query if needed"
  ]
}}
```

## Examples
- User: "I want to know about the relationship between gut microbiome and Parkinson's disease"
- Assistant: {{"queries": ["How does the gut microbiome influence or relate to Parkinson's disease pathology, symptoms, or progression?"]}}

- User: "Recent advances in CRISPR gene editing for cystic fibrosis"
- Assistant: {{"queries": [
    "What are the latest developments in using CRISPR gene editing technology to treat or cure cystic fibrosis?"}}
    "How have recent CRISPR gene editing studies impacted the treatment of cystic fibrosis?"
]}}
""",
            description="Prompt for Mistral AI to generate human-readable reranking queries"
        )
        pass

    def __init__(self):
        """Initialize the Tool."""
        self.citation = (
            False  # Disable built-in citations as we'll handle them ourselves
        )
        self.valves = self.Valves()  # Initialize admin valves
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.europepmc_base_url = (
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        )

    async def _generate_queries_with_mistral(
        self, user_query: str, api_key: str, num_queries: int,
        max_retries: int = 3, user_template: str = None, system_message: str = None
    ) -> Tuple[List[str], List[str]]:
        """
        Generate a list of search queries using Mistral AI based on user input.

        :param user_query: The user's original query
        :param api_key: Mistral AI API key
        :param num_queries: Number of queries to generate (1-5)
        :param max_retries: Maximum number of retries if format is incorrect
        :param user_template: User prompt template
        :param system_message: System prompt 
        :return: Tuple of (all_responses, queries)
        """
        # If no API key is provided, just return the original query
        if not api_key:
            return [], [user_query]

        # Ensure num_queries is within valid range
        num_queries = max(1, min(5, num_queries))

        all_responses = []

        try:
            # Import the new Mistral AI client
            from mistralai import Mistral
            client = Mistral(api_key=api_key)
            
            # Format the user prompt template with the query and num_queries
            user_message = user_template.format(
                user_query=user_query,
                num_queries=num_queries
            )

            # Create messages using the new ChatMessage class
            messages = [
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ]

            # Implement retry logic
            retry_count = 0
            while retry_count <= max_retries:
                try:
                    # Use JSON mode to ensure structured output
                    chat_response = client.chat.complete(
                        model="mistral-large-latest",
                        messages=messages,
                        response_format={"type": "json_object"},
                    )

                    response_content = chat_response.choices[0].message.content
                    all_responses.append(response_content)

                    # Parse the JSON response
                    queries_json = json.loads(response_content)
                    queries = queries_json.get("queries", [])
                    
                    # Check if we're getting a single query (for reranking)
                    if not queries and "query" in queries_json:
                        return all_responses, [queries_json["query"]]

                    # If we got valid queries, return them
                    if queries and isinstance(queries, list) and len(queries) > 0:
                        return all_responses, queries
                    
                    # If format is incorrect but we have retries left
                    if retry_count < max_retries:
                        print(f"Invalid queries format, retrying ({retry_count + 1}/{max_retries})")
                        retry_count += 1
                    else:
                        print(f"Invalid queries format after {max_retries} retries, using original query")
                        return all_responses, [user_query]

                except json.JSONDecodeError as e:
                    # JSON parsing error, retry if possible
                    if retry_count < max_retries:
                        print(f"Error parsing JSON, retrying ({retry_count + 1}/{max_retries}): {str(e)}")
                        retry_count += 1
                    else:
                        print(f"Error parsing JSON after {max_retries} retries: {str(e)}")
                        return all_responses, [user_query]
                except Exception as e:
                    # Other errors, retry if possible
                    if retry_count < max_retries:
                        print(f"Error generating queries, retrying ({retry_count + 1}/{max_retries}): {str(e)}")
                        retry_count += 1
                    else:
                        print(f"Error generating queries after {max_retries} retries: {str(e)}")
                        return all_responses, [user_query]

        except Exception as e:
            print(f"Error using Mistral AI: {str(e)}")
            # Fallback to original query
            return all_responses, [user_query]

    def _chunk_text_by_words(self, text: str, pmid: str, chunk_size: int, chunk_overlap: int) -> List[Dict]:
        """
        Split article text into overlapping chunks with unique IDs based on word count.
        
        :param text: The text to chunk (title + abstract)
        :param pmid: The PMID of the article
        :param chunk_size: Number of words per chunk
        :param chunk_overlap: Number of words to overlap between chunks
        :return: List of dicts with 'id' and 'text' keys
        """
        # Split text into words
        words = text.split()
        
        # If text is shorter than chunk_size, return it as a single chunk
        if len(words) <= chunk_size:
            return [{"id": f"{pmid}-0", "text": text}]
        
        chunks = []
        stride = chunk_size - chunk_overlap
        
        # Create chunks with overlap
        for i in range(0, len(words), stride):
            # Get chunk of words
            chunk_words = words[i:i + chunk_size]
            
            # Skip if we have less than half the chunk size (for the last chunk)
            if len(chunk_words) < chunk_size // 2:
                continue
                
            # Join words back into text
            chunk_text = " ".join(chunk_words)
            
            # Create unique ID for this chunk
            chunk_id = f"{pmid}-{len(chunks)}"
            
            chunks.append({"id": chunk_id, "text": chunk_text})
        
        return chunks

    async def _rerank_with_cohere(
        self, 
        query: str, 
        all_articles: List[Dict], 
        api_key: str, 
        chunk_size: int, 
        chunk_overlap: int,
        top_n: int,
        num_queries_rerank: int,
        rerank_query_prompt: str,
        mistral_api_key: str,
        max_retries: int,
        __event_emitter__ = None
    ) -> List[Dict]:
        """
        Rerank articles using Cohere's rerank API.

        :param query: The original user query
        :param all_articles: List of all articles to rerank
        :param api_key: Cohere API key
        :param chunk_size: Size of text chunks in words
        :param chunk_overlap: Overlap between chunks in words
        :param top_n: Number of top results to keep per query
        :param num_queries_rerank: Number of reranking queries to generate
        :param rerank_query_prompt: Prompt for generating reranking queries
        :param mistral_api_key: Mistral API key for generating queries
        :param max_retries: Maximum number of retries for query generation
        :param __event_emitter__: EventEmitter for status updates
        :return: List of filtered articles
        """
        # Create a dictionary to map PMIDs to articles for easy lookup
        pmid_to_article = {article.get("pmid", ""): article for article in all_articles}
        emitter = EventEmitter(__event_emitter__)
        
        # Generate multiple reranking queries with Mistral
        if emitter:
            await emitter.emit("Generating human-readable queries for reranking")
        
        num_queries_rerank = min(3, max(1, num_queries_rerank))
        rerank_responses, rerank_queries = await self._generate_queries_with_mistral(
            query, mistral_api_key, num_queries_rerank, max_retries, rerank_query_prompt, 
            "You are a helpful assistant that generates natural language queries for scientific article reranking."
        )
        
        # If no queries were generated, use the original query
        if not rerank_queries:
            rerank_queries = [query]
        
        # Prepare documents for reranking
        all_chunks = []
        ordered_chunk_ids = []
        
        for article in all_articles:
            pmid = article.get("pmid", "")
            title = article.get("title", "No title available")
            abstract = article.get("abstractText", "No abstract available")
            
            # Combine title and abstract for chunking
            full_text = f"{title}\n\n{abstract}"
            
            # Chunk the text
            chunks = self._chunk_text_by_words(full_text, pmid, chunk_size, chunk_overlap)
            
            # Add chunks to the list
            for chunk in chunks:
                all_chunks.append(chunk)
                ordered_chunk_ids.append(chunk["id"])
        
        # If no chunks, return all articles with no relevance scores
        if not all_chunks:
            return all_articles, {}
        
        # Initialize combined relevance scores dictionary
        combined_pmid_relevance = {}
        all_dfs = []
        
        try:
            import cohere
            
            # Initialize Cohere client
            co = cohere.ClientV2(api_key=api_key)
            
            # Process each reranking query
            for i, rerank_query in enumerate(rerank_queries):
                if emitter:
                    await emitter.emit(f"Reranking with query {i+1}/{len(rerank_queries)}: {rerank_query}")
                                    
                # Call Cohere rerank API
                response = co.rerank(
                    model="rerank-v3.5",
                    query=rerank_query,
                    documents=all_chunks,
                    top_n=top_n,
                )
                
                # Process reranking results for this query
                query_pmid_relevance = self._process_rerank_results(response, ordered_chunk_ids)
                # Update combined relevance scores
                for pmid, relevance in query_pmid_relevance.items():
                    if pmid in combined_pmid_relevance.keys():
                        old_relevance = combined_pmid_relevance[pmid]
                    if (not pmid in combined_pmid_relevance.keys()) or (relevance > old_relevance):
                        combined_pmid_relevance[pmid] = relevance

            
            # Create a filtered list of results based on combined reranking
            sorted_combined_pmids = sorted(combined_pmid_relevance.items(), key=lambda x: x[1], reverse=True)
            filtered_articles = []
            for pmid, relevance in sorted_combined_pmids:
                article = pmid_to_article.get(pmid)
                if article:
                    article['relevance'] = relevance
                    filtered_articles.append(article)
            return filtered_articles
            
        except Exception as e:
            print(f"Error in reranking process: {str(e)}")
            # Return all articles with no relevance scores on error
            return all_articles

    def _process_rerank_results(self, rerank_results: Dict, ordered_chunk_ids: List[str]) -> Dict[str, float]:
        """
        Process reranking results to get PMIDs and relevance scores.
        
        :param rerank_results: Results from Cohere rerank
        :param ordered_chunk_ids: List of chunk IDs in the same order as the documents sent to rerank
        :return: List of Dict with pmid and relevance score
        """
        # Extract PMIDs and relevance scores
        pmid_relevance = {}
        
        if "results" in rerank_results:
            for result in rerank_results["results"]:
                chunk_id = ordered_chunk_ids[result["index"]]
                pmid = chunk_id.split("-")[0]
                relevance = result["relevance_score"]
                
                if pmid not in pmid_relevance or relevance > pmid_relevance[pmid]:
                    pmid_relevance[pmid] = relevance
                
        return pmid_relevance

    async def _fetch_europepmc_results(
        self, query: str, result_type: str, max_results: int
    ) -> Dict:
        """
        Fetch results from EuropePMC API.

        :param query: Search query
        :param result_type: Result type (core, lite, etc.)
        :param max_results: Maximum number of results to return
        :return: Dictionary with search results
        """
        params = {
            "query": query,
            "format": "json",
            "resultType": result_type,
            "pageSize": max_results,
        }

        try:
            response = requests.get(
                self.europepmc_base_url, params=params, headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching results from EuropePMC: {str(e)}")
            return {"error": str(e)}

    async def search_europepmc(
        self, query: str, __user__: dict = None, __event_emitter__=None
    ) -> str:
        """
        Search EuropePMC for scientific articles based on a query, with Mistral AI query enhancement
        and optional Cohere reranking.

        :param query: The search query
        :return: Formatted results from EuropePMC
        """
        # Get admin and user valves
        user_valves = (
            __user__["valves"]
            if __user__ and "valves" in __user__
            else self.UserValves()
        )
        mistral_api_key = self.valves.mistral_api_key  # Get API key from admin valves
        num_queries = user_valves.num_queries
        max_results = user_valves.max_results
        result_type = user_valves.result_type
        user_prompt = user_valves.user_prompt
        system_prompt = user_valves.system_prompt
        citation_enabled = user_valves.citation_enabled
        
        # Get reranking parameters
        use_reranking = user_valves.use_reranking
        cohere_api_key = self.valves.cohere_api_key
        chunk_size = user_valves.chunk_size
        chunk_overlap = user_valves.chunk_overlap
        top_n = user_valves.top_n
        rerank_query_prompt = user_valves.rerank_query_prompt
        
        # Initialize the event emitter
        emitter = EventEmitter(__event_emitter__)
        
        # Emit status update
        await emitter.emit(f"Processing query: {query}")

        # Generate queries with Mistral AI
        await emitter.emit("Generating optimized search queries with Mistral AI")
        
        # Initialize the details content
        details_content = f"""<details>
<summary>EuropePMC advanced search details</summary>
### Tool User Query
```
{query}
```
### Mistral AI Generations
"""
        
        max_retries = user_valves.max_retries
        all_responses, queries = await self._generate_queries_with_mistral(
            query, mistral_api_key, num_queries, max_retries, user_prompt, system_prompt
        )
        num = 1
        for response in all_responses:
            details_content += f"- Try {num}:\n"
            details_content += f"```\n{response}\n```\n"
            num += 1
        details_content += "### Optimized Queries\n"
        # Add generated queries to the details content
        for q in queries:
            details_content += f"- {q}\n"

        # Add EuropePMC searches section
        details_content += "\n### EuropePMC searches\n"
        
        # Fetch results for each query
        all_results = []
        all_pmids = []
        
        # Dictionary to store all articles by PMID for later reranking
        pmid_to_article = {}
        
        for i, search_query in enumerate(queries):
            await emitter.emit(
                f"Searching EuropePMC for query {i+1}/{len(queries)}: {search_query}"
            )
            
            try:
                results = await self._fetch_europepmc_results(
                    search_query, result_type, max_results
                )
                
                # Count articles found for this query
                article_count = 0
                if "resultList" in results and "result" in results["resultList"]:
                    articles = results["resultList"]["result"]

                    # Process each article
                    for article in articles:
                        # Extract PMID
                        pmid = article.get("pmid", "")
                        if pmid not in all_pmids:
                            all_pmids.append(pmid)
                            all_results.append(article)
                            pmid_to_article[pmid] = article
                            article_count += 1
                        else:
                            continue
                
                # Add search result to the details content
                details_content += f"- Query {i+1}: `{search_query}` - Found {article_count} new articles\n"
                
            except Exception as e:
                await emitter.emit(
                    status="warning",
                    description=f"Error fetching results for query '{search_query}': {str(e)}",
                    done=False,
                )
                
                # Add error message to the details content
                details_content += f"- Query {i+1}: \"{search_query}\" - Error: {str(e)}\n"

        # Add summary to the details content
        details_content += f"\n### Summary\nFound {len(all_pmids)} unique articles from {len(queries)} queries\n"
        
        # Prepare filtered results and relevance scores
        filtered_results = all_results
        
        # Reranking section
        if use_reranking and cohere_api_key and all_results:
            await emitter.emit("Preparing documents for reranking with Cohere")
            details_content += "\n### Cohere Reranking\n"
            
            # Use the refactored reranking function
            filtered_results = await self._rerank_with_cohere(
                query=query,
                all_articles=all_results,
                api_key=cohere_api_key,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_n=top_n,
                num_queries_rerank=user_valves.num_queries_rerank,
                rerank_query_prompt=rerank_query_prompt,
                mistral_api_key=mistral_api_key,
                max_retries=max_retries,
                __event_emitter__=__event_emitter__
            )
            
            if len(filtered_results) < len(all_results):
                details_content += f"- Reranking filtered results from {len(all_results)} to {len(filtered_results)} articles\n"
            else:
                details_content += "- No filtering applied, using all results\n"
        else:
            # No reranking, use all results
            if not use_reranking:
                details_content += "- Reranking disabled, using all results\n"
            elif not cohere_api_key:
                details_content += "- Cohere API key not provided, using all results\n"
        
        # Close the details tag
        details_content += "</details>"
        
        # Emit the details content
        await emitter.emit_message(details_content)
        
        # Emit citations for articles
        if citation_enabled and __event_emitter__:
            await emitter.emit("Emitting citations for articles")
            for article in filtered_results:
                await emitter.emit_citation(article)
        
        # Format results
        formatted_results = [self._format_results(x) for x in filtered_results]

        # Emit final status
        await emitter.emit(
            status="complete",
            description=f"Found {len(all_results)} unique articles from EuropePMC",
            done=True,
        )

        return formatted_results
    
    def _format_results(self, article: Dict) -> str:
        """
        Format the results into a readable string.

        :param articles: List of article dictionaries
        :return: Formatted string with article information
        """
        formatted_text = ""
        if not article:
            return "No results found."
        title = article.get("title", "No title available")
        authors = article.get("authorString", "Unknown authors")
        journal = article.get("journalTitle", "Unknown journal")
        year = article.get("pubYear", "Unknown year")
        pmid = article.get("pmid", "")
        doi = article.get("doi", "")
        abstract = article.get("abstractText", "No abstract available")
        formatted_text += f"## **Title**: {title}\n"
        formatted_text += f"### **Authors**: {authors}\n"
        formatted_text += f"### **Journal**: {journal}, {year}\n"
        formatted_text += f"### **YEAR**: {year}\n"
        formatted_text += f"### **PMID**: {pmid}\n"
        formatted_text += f"### **DOI**: {doi}\n"
        formatted_text += f"### **ABSTRACT**: \n{abstract}\n"
        return formatted_text
