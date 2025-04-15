"""
title: EuropePMC Fetcher
author: jfouret
description: Fetches scientific articles from EuropePMC API with Mistral AI query generation
version: 0.1.3
license: MIT
requirements: requests, asyncio, mistralai
"""

import requests
import asyncio
import json
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Callable, Any


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
    ) -> List[str]:
        """
        Generate a list of search queries using Mistral AI based on user input.

        :param user_query: The user's original query
        :param api_key: Mistral AI API key
        :param num_queries: Number of queries to generate (1-5)
        :param max_retries: Maximum number of retries if format is incorrect
        :param user_template: User prompt template
        :param system_message: System prompt 
        :return: List of generated search queries
        """
        # If no API key is provided, just return the original query
        if not api_key:
            return [user_query]

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
        Search EuropePMC for scientific articles based on a query, with Mistral AI query enhancement.

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

                    # Emit citations for each article
                    if citation_enabled and __event_emitter__:
                        for article in articles:
                            # Extract title and abstract
                            pmid = article.get("pmid", "")
                            if pmid not in all_pmids:
                                all_pmids.append(pmid)
                                all_results.append(article)
                                article_count += 1
                            else:
                                continue
                            title = article.get("title", "No title available")

                            # The abstract might not be in the lite result type
                            # In a real implementation, you might need to make additional API calls
                            abstract = article.get(
                                "abstractText", "No abstract available"
                            )

                            # Create citation content
                            content = f"# {title}\n\n{abstract}"

                            pmid = article.get("pmid", "")

                            # Create metadata
                            metadata = {
                                "pmid": pmid,
                                "doi": article.get("doi", ""),
                                "authors": article.get("authorString", ""),
                                "journal": article.get("journalTitle", ""),
                                "publication_date": article.get(
                                    "firstPublicationDate", ""
                                ),
                                "source": f"EuropePMC:{pmid}",
                            }

                            # Emit citation
                            await __event_emitter__(
                                {
                                    "type": "citation",
                                    "data": {
                                        "document": [content],
                                        "metadata": [metadata],
                                        "source": {
                                            "name": title,
                                            "url": f"https://europepmc.org/article/MED/{pmid}",
                                        },
                                    },
                                }
                            )
                
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
        
        # Close the details tag
        details_content += "</details>"
        
        # Emit the details content
        await emitter.emit_message(details_content)

        # Format results
        formatted_results = self._format_results(all_results)

        # Emit final status
        await emitter.emit(
            status="complete",
            description=f"Found {len(all_results)} unique articles from EuropePMC",
            done=True,
        )

        return formatted_results

    def _format_results(self, articles: List[Dict]) -> str:
        """
        Format the results into a readable string.

        :param articles: List of article dictionaries
        :return: Formatted string with article information
        """
        if not articles:
            return "No results found."

        formatted_text = f"# EuropePMC Search Results\n\n"
        formatted_text += f"Found {len(articles)} articles.\n\n"

        for i, article in enumerate(articles):
            title = article.get("title", "No title available")
            authors = article.get("authorString", "Unknown authors")
            journal = article.get("journalTitle", "Unknown journal")
            year = article.get("pubYear", "Unknown year")
            pmid = article.get("pmid", "")
            doi = article.get("doi", "")

            formatted_text += f"## {i+1}. {title}\n\n"
            formatted_text += f"**Authors**: {authors}\n\n"
            formatted_text += f"**Journal**: {journal}, {year}\n\n"

            if pmid:
                formatted_text += f"**PMID**: {pmid}\n\n"
            if doi:
                formatted_text += f"**DOI**: {doi}\n\n"

            # Add abstract if available
            abstract = article.get("abstractText", "")
            if abstract:
                formatted_text += f"**Abstract**: {abstract}\n\n"

            formatted_text += "---\n\n"

        return formatted_text
