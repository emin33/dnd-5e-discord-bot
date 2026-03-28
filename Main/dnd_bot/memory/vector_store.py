"""Vector store for campaign knowledge RAG using ChromaDB."""

from pathlib import Path
from typing import Optional
import structlog

logger = structlog.get_logger()


class VectorStore:
    """
    ChromaDB-based vector store for campaign knowledge.

    Stores embeddings for:
    - Session summaries
    - NPC details
    - Location descriptions
    - Plot points and events
    - Player decisions and consequences
    """

    def __init__(self, persist_directory: str = "./data/chroma"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._client = None
        self._collection = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazily initialize ChromaDB."""
        if self._initialized:
            return

        try:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(self.persist_directory),
                anonymized_telemetry=False,
            ))

            self._initialized = True
            logger.info("chromadb_initialized", path=str(self.persist_directory))

        except ImportError:
            logger.warning(
                "chromadb_not_installed",
                message="ChromaDB not installed. Memory RAG disabled.",
            )
        except Exception as e:
            logger.error("chromadb_init_failed", error=str(e))

    def _get_collection(self, campaign_id: str):
        """Get or create collection for a campaign."""
        self._ensure_initialized()
        if not self._client:
            return None

        collection_name = f"campaign_{campaign_id}"
        return self._client.get_or_create_collection(
            name=collection_name,
            metadata={"campaign_id": campaign_id},
        )

    def add_memory(
        self,
        campaign_id: str,
        memory_id: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Add a memory to the vector store."""
        collection = self._get_collection(campaign_id)
        if not collection:
            return False

        try:
            collection.add(
                ids=[memory_id],
                documents=[content],
                metadatas=[metadata or {}],
            )
            logger.info(
                "memory_added",
                campaign_id=campaign_id,
                memory_id=memory_id,
            )
            return True
        except Exception as e:
            logger.error(
                "memory_add_failed",
                campaign_id=campaign_id,
                memory_id=memory_id,
                error=str(e),
            )
            return False

    def update_memory(
        self,
        campaign_id: str,
        memory_id: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Update an existing memory."""
        collection = self._get_collection(campaign_id)
        if not collection:
            return False

        try:
            collection.update(
                ids=[memory_id],
                documents=[content],
                metadatas=[metadata or {}],
            )
            return True
        except Exception as e:
            logger.error(
                "memory_update_failed",
                campaign_id=campaign_id,
                memory_id=memory_id,
                error=str(e),
            )
            return False

    def delete_memory(self, campaign_id: str, memory_id: str) -> bool:
        """Delete a memory from the store."""
        collection = self._get_collection(campaign_id)
        if not collection:
            return False

        try:
            collection.delete(ids=[memory_id])
            return True
        except Exception as e:
            logger.error(
                "memory_delete_failed",
                campaign_id=campaign_id,
                memory_id=memory_id,
                error=str(e),
            )
            return False

    def search(
        self,
        campaign_id: str,
        query: str,
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Search for relevant memories using semantic similarity.

        Returns list of dicts with 'id', 'content', 'distance', 'metadata'.
        """
        collection = self._get_collection(campaign_id)
        if not collection:
            return []

        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
            )

            memories = []
            if results and results.get("ids"):
                ids = results["ids"][0]
                documents = results["documents"][0] if results.get("documents") else []
                distances = results["distances"][0] if results.get("distances") else []
                metadatas = results["metadatas"][0] if results.get("metadatas") else []

                for i, memory_id in enumerate(ids):
                    memories.append({
                        "id": memory_id,
                        "content": documents[i] if i < len(documents) else "",
                        "distance": distances[i] if i < len(distances) else 0,
                        "metadata": metadatas[i] if i < len(metadatas) else {},
                    })

            return memories

        except Exception as e:
            logger.error(
                "memory_search_failed",
                campaign_id=campaign_id,
                query=query[:50],
                error=str(e),
            )
            return []

    def get_by_type(
        self,
        campaign_id: str,
        memory_type: str,
        limit: int = 10,
    ) -> list[dict]:
        """Get memories by type (npc, location, event, etc.)."""
        return self.search(
            campaign_id=campaign_id,
            query="",  # Empty query for type-based retrieval
            n_results=limit,
            where={"type": memory_type},
        )

    def add_session_summary(
        self,
        campaign_id: str,
        session_id: str,
        summary: str,
        key_events: list[str],
    ) -> bool:
        """Add a session summary to the store."""
        return self.add_memory(
            campaign_id=campaign_id,
            memory_id=f"session_{session_id}",
            content=f"Session Summary:\n{summary}\n\nKey Events:\n" + "\n".join(f"- {e}" for e in key_events),
            metadata={
                "type": "session",
                "session_id": session_id,
            },
        )

    def add_npc(
        self,
        campaign_id: str,
        npc_id: str,
        name: str,
        description: str,
        notes: str = "",
    ) -> bool:
        """Add an NPC to the store."""
        content = f"NPC: {name}\n{description}"
        if notes:
            content += f"\n\nNotes: {notes}"

        return self.add_memory(
            campaign_id=campaign_id,
            memory_id=f"npc_{npc_id}",
            content=content,
            metadata={
                "type": "npc",
                "npc_id": npc_id,
                "name": name,
            },
        )

    def add_location(
        self,
        campaign_id: str,
        location_id: str,
        name: str,
        description: str,
    ) -> bool:
        """Add a location to the store."""
        return self.add_memory(
            campaign_id=campaign_id,
            memory_id=f"location_{location_id}",
            content=f"Location: {name}\n{description}",
            metadata={
                "type": "location",
                "location_id": location_id,
                "name": name,
            },
        )

    def add_event(
        self,
        campaign_id: str,
        event_id: str,
        description: str,
        session_id: Optional[str] = None,
    ) -> bool:
        """Add a significant event to the store."""
        metadata = {"type": "event", "event_id": event_id}
        if session_id:
            metadata["session_id"] = session_id

        return self.add_memory(
            campaign_id=campaign_id,
            memory_id=f"event_{event_id}",
            content=description,
            metadata=metadata,
        )

    def recall_for_context(
        self,
        campaign_id: str,
        current_situation: str,
        max_results: int = 3,
    ) -> str:
        """
        Recall relevant memories for the current context.

        Returns formatted string for LLM context.
        """
        memories = self.search(
            campaign_id=campaign_id,
            query=current_situation,
            n_results=max_results,
        )

        if not memories:
            return ""

        lines = ["<recalled_memories>", "Relevant campaign history:"]
        for memory in memories:
            lines.append(f"\n{memory['content']}")
        lines.append("</recalled_memories>")

        return "\n".join(lines)

    def persist(self) -> None:
        """Persist the database to disk."""
        if self._client:
            try:
                self._client.persist()
            except Exception as e:
                logger.error("chromadb_persist_failed", error=str(e))


# Singleton instance
_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get the singleton vector store."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
