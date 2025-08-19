"""
title: EuropePMC Fetcher
author: jfouret
description: Fetches scientific articles from EuropePMC API using a LangGraph RAG workflow with LLM query generation and Cohere reranking.
version: 0.3.0
required_open_webui_version: 0.6.20
license: MIT
requirements: requests, asyncio, mistralai, cohere, pandas, langchain, langgraph, langchain-cohere, langchain-mistralai, langchain-community, faiss-cpu, langchain_openai, tiktoken, langchain-europe-pmc
"""

import asyncio
import json
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr
from typing import List, Dict, Callable, Any, Literal
from typing_extensions import TypedDict

# Langchain and LangGraph imports
from langchain_core.documents import Document
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_europe_pmc.retrievers import EuropePMCRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


class EventEmitter(BaseModel):
    _emitter: Callable[[dict], Any] = PrivateAttr(default=None)

    def __init__(self, event_emitter: Callable[[dict], Any] = None, **kwargs):
        super().__init__(**kwargs)
        self._emitter = event_emitter

    async def emit(self, description="Unknown State", status="in_progress", done=False):
        """Emit a status update."""
        if self._emitter:
            await self._emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )
    async def emit_citation(self, article, citation_id: int):
        """Emit a citation."""
        pmid = article.get("pmid", "")
        title = article.get("title", "No title available")
        abstract = article.get("abstractText", "No abstract available")
        content = f"# {title}\n\n{abstract}"

        metadata = {
            "pmid": pmid,
            "doi": article.get("doi", ""),
            "authors": article.get("authorString", ""),
            "journal": article.get("journalTitle", ""),
            "relevance": article.get("relevance", None),
            "publication_date": article.get("firstPublicationDate", ""),
            "source": f"EuropePMC:{pmid}",
        }
        if self._emitter:
            await self._emitter(
                {
                    "type": "citation",
                    "data": {
                        "id": citation_id,
                        "document": [content],
                        "metadata": [metadata],
                        "source": {
                            "id": citation_id,
                            "name": title,
                            "url": f"https://europepmc.org/article/MED/{pmid}",
                        },
                        "distances": (
                            [metadata["relevance"]]
                            if metadata["relevance"] is not None
                            else []
                        ),
                    },
                }
            )

    async def emit_message(self, content):
        """Emit a message with the given content."""
        if self._emitter:
            await self._emitter(
                {
                    "type": "message",
                    "data": {"content": content},
                }
            )


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: user question
        queries: EuropePMC queries
        documents: The list of retrieved documents
        answer: The final answer
        details: A markdown string with details of the run
        event_emitter: The event emitter instance
        user_valves: User-configurable settings
    """

    question: str
    queries: List[str]
    documents: List[Document]
    answer: str
    details: str
    event_emitter: EventEmitter
    user_valves: "Tools.UserValves"


class Tools:
    class Valves(BaseModel):
        mistral_api_key: str = Field(
            default="",
            description="Mistral AI API key for query generation and answering (admin only)",
        )
        cohere_api_key: str = Field(
            default="",
            description="Cohere API key for reranking (required if use_reranking is True)",
        )
        openai_api_key: str = Field(
            default="",
            description="OpenAI API key for query generation and answering (admin only)",
        )

    class UserValves(BaseModel):
        n_retry: int = Field(
            default=3,
            description="Number of retries if the query generation fails to produce the correct format.",
        )
        n_search: int = Field(
            default=3,
            description="Number of different search queries to generate.",
        )
        max_results: int = Field(
            default=20,
            description="Maximum number of results to retrieve from EuropePMC.",
        )
        search_strategy: Literal["rerank", "semantic_search"] = Field(
            default="rerank",
            description="Refinement strategy to use: 'rerank' or 'semantic_search'.",
        )
        top_n: int = Field(
            default=10,
            description="Number of top results to keep after refinement (reranking or semantic search).",
        )
        llm_provider: Literal["openai", "mistral"] = Field(
            default="mistral",
            description="LLM provider to use for generation (mistral or openai).",
        )
        model_name: str = Field(
            default="mistral-medium-latest",
            description="Name of the model to use for generation (e.g., mistral-large-latest, gpt-4-turbo).",
        )
        cohere_rerank_model: str = Field(
            default="rerank-v3.5",
            description="Cohere model to use for reranking.",
        )
        cohere_embedding_model: str = Field(
            default="embed-v4.0",
            description="Cohere model to use for embeddings in semantic search.",
        )
        citation_enabled: bool = Field(
            default=True, description="If True, send custom citations with links"
        )

    class _DummyRetriever(BaseRetriever):
        """A dummy retriever that just returns a given list of documents."""

        docs: List[Document]

        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> List[Document]:
            return self.docs

        async def _aget_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> List[Document]:
            return self.docs

    def __init__(self):
        """Initialize the Tool."""
        self.citation = False
        self.valves = self.Valves()

    def _get_llm(self, user_valves: UserValves):
        if user_valves.llm_provider == "openai":
            if not self.valves.openai_api_key:
                raise ValueError("OpenAI API key is not set in admin valves.")
            return ChatOpenAI(
                model=user_valves.model_name,
                api_key=self.valves.openai_api_key,
                temperature=0,
                streaming=True,
            )
        elif user_valves.llm_provider == "mistral":
            if not self.valves.mistral_api_key:
                raise ValueError("Mistral API key is not set in admin valves.")
            return ChatMistralAI(
                model=user_valves.model_name,
                api_key=self.valves.mistral_api_key,
                temperature=0,
                streaming=True,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {user_valves.llm_provider}")

    # Graph Nodes
    async def _generate_query(self, state: GraphState) -> GraphState:
        event_emitter = state["event_emitter"]
        question = state["question"]
        user_valves = state["user_valves"]
        n_retry = user_valves.n_retry
        n_search = user_valves.n_search

        class EuropePMCQueries(BaseModel):
            """EuropePMC search queries."""

            queries: List[str] = Field(
                description=f"List of {n_search} different search query strings for EuropePMC."
            )

        schema = json.dumps(EuropePMCQueries.model_json_schema(), indent=2)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a specialized scientific literature search assistant for EuropePMC.
Your task is to transform a user's research question into {n_search} different, optimized search query strings using EuropePMC's advanced search syntax. A researcher would try multiple queries to find the most relevant papers, your role is to emulate this behaviour.
- Use quotation marks for exact phrases: "Small cell lung cancer"
- Use Boolean operators (AND, OR, NOT) and group with parentheses.
- Use field-specific searches: TITLE:"term", AUTH:"Name", PUB_YEAR:YYYY
- Use wildcards (*).
- You must generate exactly {n_search} queries.
- You must format your response as a JSON object that respects the following JSON schema:
{schema}""",
                ),
                ("user", "Research Question: {question}"),
            ]
        )

        llm = self._get_llm(user_valves)
        structured_llm = llm.with_structured_output(EuropePMCQueries, method="json_mode")
        query_generation_chain = prompt.partial(schema=schema) | structured_llm

        queries = []
        last_error = "No queries were generated."

        for attempt in range(n_retry):
            await event_emitter.emit(
                f"Generating optimized search queries (Attempt {attempt + 1}/{n_retry})..."
            )
            try:
                response = await query_generation_chain.ainvoke(
                    {"question": question, "n_search": n_search}
                )
                if isinstance(response, EuropePMCQueries) and len(
                    response.queries
                ) == n_search:
                    queries = response.queries
                    fqueries = "\n- +".join(queries)
                    state[
                        "details"
                    ] += f"### Generated Queries\n- {fqueries}\n"
                    break  # Success
                else:
                    last_error = f"LLM failed to generate {n_search} queries. Response: {response}"
                    await event_emitter.emit(last_error, status="error")

            except Exception as e:
                last_error = str(e)
                await event_emitter.emit(
                    f"Error on query generation attempt {attempt + 1}: {last_error}",
                    status="error",
                )

            if not queries and attempt < n_retry - 1:
                await asyncio.sleep(1)  # wait before retrying if not successful

        if not queries:
            await event_emitter.emit(
                f"Query generation failed after {n_retry} attempts. Using original question.",
                status="error",
            )
            queries = [question]
            state[
                "details"
            ] += f"### Query Generation Failed\n- Final error: {last_error}\n- Using original question as query: `{queries[0]}`\n"

        state["queries"] = queries
        return state

    async def _retrieve(self, state: GraphState) -> GraphState:
        event_emitter = state["event_emitter"]
        queries = state["queries"]
        await event_emitter.emit(
            f"Retrieving articles for {len(queries)} queries from EuropePMC..."
        )
        retriever = EuropePMCRetriever(max_k=state["user_valves"].max_results)

        all_documents = {}  # Use a dict to handle duplicates based on pmid
        state["details"] += "\n### Retrieval\n"

        for i, query in enumerate(queries):
            await event_emitter.emit(
                f"Running search for query {i+1}/{len(queries)}: `{query}`"
            )
            try:
                documents = await retriever.ainvoke(query)
                for doc in documents:
                    pmid = doc.metadata.get("pmid")
                    if pmid and pmid not in all_documents:
                        all_documents[pmid] = doc
                state[
                    "details"
                ] += f"- Query `{query}` found {len(documents)} documents.\n"

            except Exception as e:
                await event_emitter.emit(
                    f"Error retrieving documents for query '{query}': {e}",
                    status="error",
                )
                state["details"] += f"- Retrieval Failed for query `{query}`: {e}\n"

        unique_documents = list(all_documents.values())
        state[
            "details"
        ] += f"- Found {len(unique_documents)} unique documents in total.\n"
        state["documents"] = unique_documents
        return state

    async def _rerank(self, state: GraphState) -> GraphState:
        event_emitter = state["event_emitter"]
        await event_emitter.emit("Reranking documents with Cohere...")
        compressor = CohereRerank(
            cohere_api_key=self.valves.cohere_api_key,
            top_n=state["user_valves"].top_n,
            model=state["user_valves"].cohere_rerank_model,
        )
        retriever = self._DummyRetriever(docs=state["documents"])
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
        try:
            reranked_docs = await compression_retriever.ainvoke(state["question"])
            state[
                "details"
            ] += f"\n### Reranking\n- Reranked to {len(reranked_docs)} documents.\n"
            state["documents"] = reranked_docs
        except Exception as e:
            await event_emitter.emit(f"Error reranking documents: {e}", status="error")
            state["details"] += f"\n### Reranking Failed\n- Error: {e}\n"
        return state

    def _decide_refinement_strategy(self, state: GraphState) -> str:
        if not self.valves.cohere_api_key or not state["documents"]:
            return "generate_answer"

        strategy = state["user_valves"].search_strategy
        if strategy == "rerank":
            return "rerank"
        elif strategy == "semantic_search":
            return "semantic_search"
        return "generate_answer"

    async def _semantic_search(self, state: GraphState) -> GraphState:
        event_emitter = state["event_emitter"]
        await event_emitter.emit("Performing semantic search with Cohere embeddings...")
        question = state["question"]
        documents = state["documents"]
        user_valves = state["user_valves"]

        try:
            embeddings = CohereEmbeddings(
                cohere_api_key=self.valves.cohere_api_key,
                model=user_valves.cohere_embedding_model,
            )
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1024, chunk_overlap=100
            )
            docs_for_vectorstore = text_splitter.split_documents(documents)

            vectorstore = await FAISS.afrom_documents(docs_for_vectorstore, embeddings)
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": user_valves.top_n}
            )

            semantically_searched_docs = await retriever.ainvoke(question)

            state[
                "details"
            ] += f"\n### Semantic Search\n- Found {len(semantically_searched_docs)} semantically similar documents.\n"
            state["documents"] = semantically_searched_docs
        except Exception as e:
            await event_emitter.emit(
                f"Error during semantic search: {e}", status="error"
            )
            state["details"] += f"\n### Semantic Search Failed\n- Error: {e}\n"
        return state

    async def _generate_answer(self, state: GraphState) -> GraphState:
        event_emitter = state["event_emitter"]
        await event_emitter.emit("Preparing context for the main model...")
        if not state["documents"]:
            state["answer"] = "No relevant documents were found to answer the question."
            state["details"] += "\n### Answer Generation\n- No documents to process.\n"
            return state

        prompt_template = """### Task:
Respond to the user query using the provided context, incorporating inline citations in the format [id] **only when the <source> tag includes an explicit id attribute** (e.g., <source id="1">).
### Guidelines:
- If you don't know the answer, clearly state that.
- If uncertain, ask the user for clarification.
- Respond in the same language as the user's query.
- If the context is unreadable or of poor quality, inform the user and provide the best possible answer.
- If the answer isn't present in the context but you possess the knowledge, explain this to the user and provide the answer using your own understanding.
- **Only include inline citations using [id] (e.g., [1], [2]) when the <source> tag includes an id attribute.**
- Do not cite if the <source> tag does not contain an id attribute.
- Do not use XML tags in your response.
- Ensure citations are concise and directly related to the information provided.
### Example of Citation:
If the user asks about a specific topic and the information is found in a source with a provided id attribute, the response should include the citation like in the following example:
* "According to the study, the proposed method increases efficiency by 20% [1]."
### Output:
Provide a clear and direct response to the user's query, including inline citations in the format [id] only when the <source> tag with id attribute is present in the context.
<user_query>
{query}
</user_query>
"""
        context_parts = []
        for i, doc in enumerate(state["documents"]):
            pmid = doc.metadata.get("pmid", "N/A")
            if pmid != "N/A":
                context_part = f'<source id="{i + 1}">\n'
                context_part += f"Title: {doc.metadata.get('title', 'N/A')}\n"
                context_part += f"Abstract: {doc.page_content}\n"
                context_part += "</source>"
                context_parts.append(context_part)

        context = "\n\n".join(context_parts)
        query = state["question"]

        final_prompt = (
            prompt_template.format(query=query) + f"\n\n### Context:\n{context}"
        )

        state["answer"] = final_prompt
        state[
            "details"
        ] += "\n### Prompt Generation\n- Successfully generated a prompt for the main model.\n"
        return state

    async def search_europepmc(
        self, query: str, __user__: dict = None, __event_emitter__=None
    ) -> str:
        """
        Search EuropePMC for scientific articles using a LangGraph RAG workflow.

        :param query: The user's research question.
        :return: A generated answer based on retrieved scientific articles.
        """
        user_valves = __user__.get("valves", self.UserValves())
        emitter = EventEmitter(__event_emitter__)

        workflow = StateGraph(GraphState)
        workflow.add_node("generate_query", self._generate_query)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("rerank", self._rerank)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("semantic_search", self._semantic_search)

        workflow.set_entry_point("generate_query")
        workflow.add_edge("generate_query", "retrieve")
        workflow.add_conditional_edges(
            "retrieve",
            self._decide_refinement_strategy,
            {
                "rerank": "rerank",
                "semantic_search": "semantic_search",
                "generate_answer": "generate_answer",
            },
        )
        workflow.add_edge("rerank", "generate_answer")
        workflow.add_edge("semantic_search", "generate_answer")
        workflow.add_edge("generate_answer", END)

        app = workflow.compile()

        initial_state = {
            "question": query,
            "queries": [],
            "details": f"<details><summary>EuropePMC RAG Details</summary>\n### User Query\n- `{query}`\n",
            "event_emitter": emitter,
            "user_valves": user_valves,
        }

        final_state = await app.ainvoke(initial_state)

        details_content = final_state.get("details", "") + "</details>"
        await emitter.emit_message(details_content)

        if user_valves.citation_enabled and __event_emitter__:
            await emitter.emit("Emitting citations for articles")
            for i, doc in enumerate(final_state.get("documents", [])):
                article = {
                    "pmid": doc.metadata.get("pmid"),
                    "title": doc.metadata.get("title"),
                    "abstractText": doc.page_content,
                    "doi": doc.metadata.get("doi"),
                    "authorString": doc.metadata.get("authors"),
                    "journalTitle": doc.metadata.get("journal"),
                    "firstPublicationDate": doc.metadata.get("year"),
                    "relevance": doc.metadata.get("relevance_score"),
                }
                await emitter.emit_citation(article, i + 1)

        await emitter.emit(
            status="complete",
            description=f"Finished processing: {len(final_state.get('documents', []))} documents referenced.",
            done=True,
        )

        return final_state.get("answer", "No answer could be generated.")

        return final_state.get("answer", "No answer could be generated.")


