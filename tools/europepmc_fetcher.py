"""
title: EuropePMC Fetcher
author: jfouret
description: Fetches scientific articles from EuropePMC API with Mistral AI query generation
version: 0.1.1
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
        self, user_query: str, api_key: str, num_queries: int, max_retries: int = 3
    ) -> List[str]:
        """
        Generate a list of search queries using Mistral AI based on user input.

        :param user_query: The user's original query
        :param api_key: Mistral AI API key
        :param num_queries: Number of queries to generate (1-5)
        :param max_retries: Maximum number of retries if format is incorrect
        :return: List of generated search queries
        """
        # If no API key is provided, just return the original query
        if not api_key:
            return [user_query]

        # Ensure num_queries is within valid range
        num_queries = max(1, min(5, num_queries))

        try:
            # Import the new Mistral AI client
            from mistralai import Mistral
            client = Mistral(api_key=api_key)

            # Create system and user messages with detailed prompt
            system_message = ""
            user_message = f"""
## Task
Generate optimized search queries for EuropePMC based on the user's medical or scientific research question. Your queries should cover different aspects and synonyms of the topic to ensure comprehensive search results. Each query should be tailored for scientific literature search in the medical and biological domains.
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
## Example
- user: I want to know about the relationship between gut microbiome and Parkinson's disease
- assistant: {{"queries":["gut microbiome Parkinson's disease association", "intestinal microbiota neurodegeneration Parkinson", "microbiome dysbiosis Parkinson's pathophysiology", "gastrointestinal microbiome neurological disorders Parkinson's"]}}
"""

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

                    # Parse the JSON response
                    queries_json = json.loads(response_content)
                    queries = queries_json.get("queries", [])

                    # If we got valid queries, return them
                    if queries and isinstance(queries, list) and len(queries) > 0:
                        return queries
                    
                    # If format is incorrect but we have retries left
                    if retry_count < max_retries:
                        print(f"Invalid queries format, retrying ({retry_count + 1}/{max_retries})")
                        retry_count += 1
                    else:
                        print(f"Invalid queries format after {max_retries} retries, using original query")
                        return [user_query]

                except json.JSONDecodeError as e:
                    # JSON parsing error, retry if possible
                    if retry_count < max_retries:
                        print(f"Error parsing JSON, retrying ({retry_count + 1}/{max_retries}): {str(e)}")
                        retry_count += 1
                    else:
                        print(f"Error parsing JSON after {max_retries} retries: {str(e)}")
                        return [user_query]
                except Exception as e:
                    # Other errors, retry if possible
                    if retry_count < max_retries:
                        print(f"Error generating queries, retrying ({retry_count + 1}/{max_retries}): {str(e)}")
                        retry_count += 1
                    else:
                        print(f"Error generating queries after {max_retries} retries: {str(e)}")
                        return [user_query]

        except Exception as e:
            print(f"Error using Mistral AI: {str(e)}")
            # Fallback to original query
            return [user_query]

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
        citation_enabled = user_valves.citation_enabled

        emitter = EventEmitter(__event_emitter__)
        await emitter.emit(f"Processing query: {query}")

        # Generate queries with Mistral AI
        await emitter.emit("Generating optimized search queries with Mistral AI")
        try:
            max_retries = user_valves.max_retries
            queries = await self._generate_queries_with_mistral(
                query, mistral_api_key, num_queries, max_retries
            )
            
            # Emit a message with the generated queries
            if __event_emitter__ and queries:
                # Format queries as a markdown list
                queries_markdown = "> Generated Search Queries\n"
                for i, q in enumerate(queries):
                    queries_markdown += f"> {i+1}. {q}\n"
                queries_markdown += "\n"
                # Emit the message
                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {
                            "content": queries_markdown
                        },
                    }
                )
                
        except Exception as e:
            await emitter.emit(
                status="error",
                description=f"Error generating queries with Mistral AI: {str(e)}",
                done=True,
            )
            queries = [query]  # Fallback to original query

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
            except Exception as e:
                await emitter.emit(
                    status="warning",
                    description=f"Error fetching results for query '{search_query}': {str(e)}",
                    done=False,
                )

        # Format results
        formatted_results = self._format_results(all_results)

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
