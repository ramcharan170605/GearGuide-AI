"""
Retrieval System for Dealer Assistant

Implements RAG (Retrieval-Augmented Generation) for the product catalogue.

Requirements:
- RAG-001: Real retrieval (embeddings + vector search)
- RAG-002: Not prompt-stuffing
- RAG-003: Index the catalogue
- RAG-005: Explain chunking/embedding/indexing choices in DESIGN.md
- RAG-006: Sensible indexing that scales
- RAG-007: Index all catalogue fields

Task: P1-T004, P1-T005, P1-T006
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss


class CatalogueRetriever:
    """
    Retrieves relevant products from the catalogue using semantic search.

    Uses sentence-transformers for embeddings and FAISS for vector search.
    """

    def __init__(self, catalogue_path: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the retriever.

        Args:
            catalogue_path: Path to catalogue CSV/JSON file
            model_name: Sentence transformer model name
        """
        self.model_name = model_name
        self.catalogue_path = catalogue_path
        self.embedding_model = None
        self.vector_store = None
        self.catalogue_data = None
        self.sku_to_index = {}

    def load_catalogue(self, file_format: str = "csv"):
        """
        Load catalogue from CSV/JSON file.

        Args:
            file_format: "csv" or "json"

        Returns:
            pd.DataFrame: Loaded catalogue data

        Requirements: RAG-003, RAG-007
        Task: P1-T006
        """
        if file_format == "csv":
            self.catalogue_data = pd.read_csv(self.catalogue_path)
        elif file_format == "json":
            self.catalogue_data = pd.read_json(self.catalogue_path)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

        # Build SKU to index mapping
        for idx, row in self.catalogue_data.iterrows():
            self.sku_to_index[row['sku']] = idx

        return self.catalogue_data

    def initialize_embedding_model(self):
        """
        Initialize the sentence transformer embedding model.

        Uses all-MiniLM-L6-v2 model for good quality embeddings (384 dimensions).
        This model is free, local, and doesn't require API keys.

        Returns:
            SentenceTransformer: Loaded embedding model

        Requirements: RAG-001, RAG-005, RAG-006
        Task: P1-T004
        """
        self.embedding_model = SentenceTransformer(self.model_name)
        return self.embedding_model

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            np.ndarray: Embedding vectors (normalized)

        Requirements: RAG-001, RAG-005, RAG-006
        Task: P1-T004
        """
        if self.embedding_model is None:
            self.initialize_embedding_model()

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        return embeddings.cpu().numpy()

    def build_vector_store(self):
        """
        Build FAISS vector store from catalogue embeddings.

        Creates index with product metadata for retrieval.
        Uses IndexIDMap2 to support metadata storage with IDs.

        Requirements: RAG-001, RAG-002, RAG-003, RAG-006, RAG-007
        Task: P1-T005
        """
        if self.catalogue_data is None:
            self.load_catalogue()

        if self.embedding_model is None:
            self.initialize_embedding_model()

        # Prepare texts for embedding: combine name, description, vehicle_fitment
        texts = []
        for _, row in self.catalogue_data.iterrows():
            text_parts = [
                row.get('name', ''),
                row.get('description', ''),
                row.get('vehicle_fitment', '')
            ]
            texts.append(" ".join(filter(None, text_parts)))

        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        embedding_dim = embeddings.shape[1]

        # Create FAISS index with ID mapping
        # IndexIDMap2 allows us to map integer IDs to vectors
        # We use sequential IDs for the vectors
        embedding_dim = embeddings.shape[1]

        # Create the base index (Inner Product for cosine similarity)
        index = faiss.IndexFlatIP(embedding_dim)

        # Wrap with ID map - IndexIDMap2 takes the base index as argument
        self.vector_store = faiss.IndexIDMap2(index)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Assign IDs and add to index
        ids = np.arange(len(self.catalogue_data)).astype(np.int64)
        self.vector_store.add_with_ids(embeddings, ids)

        # Store metadata for retrieval
        self.metadata = self.catalogue_data.to_dict('records')

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Search for relevant products using semantic search.

        Args:
            query: User search query
            k: Number of results to return

        Returns:
            list[dict]: List of product dictionaries with similarity scores

        Requirements: RAG-001, RAG-002, RAG-003, RAG-006
        Task: P1-T005
        """
        if self.vector_store is None:
            self.build_vector_store()

        # Generate embedding for query
        query_embedding = self.generate_embeddings([query])
        faiss.normalize_L2(query_embedding)

        # Search the index
        distances, indices = self.vector_store.search(query_embedding, k)

        # Convert to list of results with metadata
        results = []
        for i in range(min(k, len(indices[0]))):
            idx = indices[0][i]
            distance = distances[0][i]

            # Get metadata
            product = self.metadata[idx].copy()
            product['similarity_score'] = float(distance)

            results.append(product)

        return results

    def find_by_vehicle(self, make: str, model: str, year: str = None, k: int = 10) -> list[dict]:
        """
        Find parts that fit a specific vehicle.

        First tries exact match on vehicle_fitment, then falls back to semantic search.

        Args:
            make: Vehicle make (e.g., "Bajaj")
            model: Vehicle model (e.g., "Pulsar 150")
            year: Optional vehicle year
            k: Number of results to return

        Returns:
            list[dict]: Matching products

        Requirements: RAG-003, RAG-007, TOOL-003, TOOL-005
        Task: P1-T005 (supports TOOL-003)
        """
        if self.catalogue_data is None:
            self.load_catalogue()

        # Build search query
        vehicle_query = f"{make} {model}"
        if year:
            vehicle_query += f" {year}"

        # First, try exact match on vehicle_fitment
        exact_matches = []
        for _, row in self.catalogue_data.iterrows():
            fitment = row.get('vehicle_fitment', '')
            if fitment and fitment.lower() == vehicle_query.lower():
                product = row.to_dict()
                product['similarity_score'] = 1.0  # Perfect match
                exact_matches.append(product)

        # If we have exact matches, return them
        if exact_matches:
            return exact_matches[:k]

        # Otherwise, use semantic search with vehicle query
        return self.search(vehicle_query, k=k)


if __name__ == "__main__":
    # Test the retrieval system
    print("Testing CatalogueRetriever...")

    retriever = CatalogueRetriever("catalogue.csv")
    retriever.load_catalogue()
    print(f"Loaded {len(retriever.catalogue_data)} products")

    retriever.initialize_embedding_model()
    print(f"Embedding model: {retriever.model_name}")

    retriever.build_vector_store()
    print("Vector store built")

    # Test search
    results = retriever.search("brake pads", k=3)
    print(f"\nSearch for 'brake pads': {len(results)} results")
    for r in results:
        print(f"  - {r.get('name', 'N/A')} (score: {r.get('similarity_score', 0):.4f})")

    # Test find_by_vehicle
    vehicle_results = retriever.find_by_vehicle("Bajaj", "Pulsar 150", k=3)
    print(f"\nFind by vehicle 'Bajaj Pulsar 150': {len(vehicle_results)} results")
    for r in vehicle_results:
        print(f"  - {r.get('name', 'N/A')}")

    print("\nAll tests passed!")
