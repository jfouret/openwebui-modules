"""
title: Web Fetcher
author: jfouret
description: Fetches content from a webpage and optionally follows links on the same domain
version: 0.1.1
license: MIT
requirements: requests, beautifulsoup4, markdownify, urllib3, asyncio
"""

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from urllib.parse import urlparse, urljoin
from pydantic import BaseModel, Field
from typing import List, Dict, Set, Optional, Union, Callable, Any
import time
import asyncio
import json

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
    class UserValves(BaseModel):
        depth_limit: int = Field(
            default=0, 
            description="Maximum depth for following links (0 means no following)"
        )
        max_pages: int = Field(
            default=10,
            description="Maximum number of pages to fetch when following links"
        )
        follow_delay: float = Field(
            default=0.5,
            description="Delay between requests in seconds when following links"
        )
        citation_links: bool = Field(
            default=True,
            description="If True, send custom citations with links"
        )
        pass

    def __init__(self):
        """Initialize the Tool."""
        self.citation = False  # Disable built-in citations as we'll handle them ourselves
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    async def fetch_webpage(self, url: str, __user__: dict = None, __event_emitter__=None) -> str:
        """
        Fetches content from a webpage and optionally follows links on the same domain.
        
        :param url: The URL to fetch content from
        :return: The webpage content converted to markdown
        """
        # Get user valves
        user_valves = __user__["valves"] if __user__ and "valves" in __user__ else self.UserValves()
        depth_limit = user_valves.depth_limit
        max_pages = user_valves.max_pages
        follow_delay = user_valves.follow_delay
        citation_links = user_valves.citation_links
        
        emitter = EventEmitter(__event_emitter__)
        await emitter.emit(f"Initiating web fetch for: {url}")
        
        # Parse the domain from the URL
        parsed_url = urlparse(url)
        base_domain = parsed_url.netloc
        
        # Initialize variables for link following
        visited_urls = set()
        results = {}
        
        # Fetch the initial page
        try:
            await emitter.emit(f"Fetching content from {url}")
            initial_content = await self._fetch_single_page(url)
            if initial_content:
                results[url] = initial_content
                visited_urls.add(url)
                
                # Emit citation for the initial page
                if citation_links and __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "citation",
                            "data": {
                                "document": [initial_content["markdown"]],
                                "metadata": [{"source": url}],
                                "source": {"name": initial_content["title"] or url},
                            },
                        }
                    )
            
            # Follow links if depth_limit > 0
            if depth_limit > 0:
                await self._follow_links(
                    url, 
                    base_domain, 
                    depth_limit, 
                    max_pages, 
                    follow_delay,
                    visited_urls, 
                    results,
                    citation_links,
                    __event_emitter__
                )
        except Exception as e:
            await emitter.emit(
                status="error",
                description=f"Error fetching content: {str(e)}",
                done=True
            )
            return f"Error fetching content from {url}: {str(e)}"
        
        # Compile the results
        if not results:
            return f"Could not fetch any content from {url}"
        
        # Create a summary of the results that includes the actual content
        summary = f"### Web Content from {url}\n\n"
        summary += "The following information was extracted directly from the website and should be used as the primary source for your response:\n\n"
        
        # Add the initial page content
        if url in results:
            summary += f"## {results[url]['title'] or 'Main Page'}\n\n"
            summary += results[url]['markdown']
            summary += "\n\n"
        
        # Add content from followed links if any
        if len(results) > 1:
            summary += f"## Additional Pages ({len(results) - 1})\n\n"
            for link_url, content in results.items():
                if link_url != url:  # Skip the initial URL
                    summary += f"### {content['title'] or link_url}\n\n"
                    summary += f"Source: {link_url}\n\n"
                    summary += content['markdown']
                    summary += "\n\n---\n\n"
        
        # Emit completion status
        await emitter.emit(
            status="complete",
            description=f"Fetched {len(results)} pages from {base_domain}",
            done=True
        )
        
        return summary

    async def _fetch_single_page(self, url: str) -> Optional[Dict]:
        """
        Fetches a single webpage and extracts its content.
        
        :param url: The URL to fetch
        :return: Dictionary with page content or None if failed
        """
        try:
            # Use a synchronous request but wrap it in an async function
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract the title
            title = soup.title.string if soup.title else None
            
            # Extract the main content (try to find main content areas)
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.body
            
            # If no specific content area found, use the whole body
            if not main_content:
                main_content = soup
            
            # Remove script and style elements
            for script in main_content(["script", "style"]):
                script.decompose()
            
            # Convert to markdown
            markdown_content = md(str(main_content))
            
            return {
                'title': title,
                'html': str(main_content),
                'markdown': markdown_content,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            }
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return None

    async def _follow_links(
        self, 
        base_url: str, 
        base_domain: str, 
        depth_limit: int, 
        max_pages: int,
        follow_delay: float,
        visited_urls: Set[str], 
        results: Dict[str, Dict],
        citation_links: bool,
        __event_emitter__=None
    ) -> None:
        """
        Recursively follows links on the same domain up to the specified depth.
        
        :param base_url: The starting URL
        :param base_domain: The domain to stay within
        :param depth_limit: Maximum depth to follow links
        :param max_pages: Maximum number of pages to fetch
        :param follow_delay: Delay between requests in seconds
        :param visited_urls: Set of already visited URLs
        :param results: Dictionary to store results
        :param citation_links: Whether to emit citations for links
        :param __event_emitter__: Event emitter for status updates
        """
        if depth_limit <= 0 or len(results) >= max_pages:
            return
        
        # Get the page content
        content = results.get(base_url)
        if not content:
            return
        
        # Parse the HTML to find links
        soup = BeautifulSoup(content['html'], 'html.parser')
        links = soup.find_all('a', href=True)
        
        # Process each link
        for link in links:
            if len(results) >= max_pages:
                break
                
            href = link['href']
            
            # Skip empty links, anchors, or non-HTTP links
            if not href or href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
                continue
            
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            
            # Parse the URL to check the domain
            parsed_url = urlparse(absolute_url)
            
            # Skip if not the same domain or already visited
            if parsed_url.netloc != base_domain or absolute_url in visited_urls:
                continue
            
            # Add to visited URLs
            visited_urls.add(absolute_url)
            
            # Emit status update
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Following link: {absolute_url}", "done": False},
                    }
                )
            
            # Add delay between requests
            await asyncio.sleep(follow_delay)
            
            # Fetch the linked page
            linked_content = await self._fetch_single_page(absolute_url)
            if linked_content:
                results[absolute_url] = linked_content
                
                # Emit citation for the linked page
                if citation_links and __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "citation",
                            "data": {
                                "document": [linked_content["markdown"]],
                                "metadata": [{"source": absolute_url}],
                                "source": {"name": linked_content["title"] or absolute_url},
                            },
                        }
                    )
                
                # Recursively follow links from this page with reduced depth
                await self._follow_links(
                    absolute_url, 
                    base_domain, 
                    depth_limit - 1, 
                    max_pages, 
                    follow_delay,
                    visited_urls, 
                    results,
                    citation_links,
                    __event_emitter__
                )
