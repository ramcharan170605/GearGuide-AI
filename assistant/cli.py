"""
CLI interface for Dealer Assistant

Provides interactive command-line interface for the LLM-driven assistant.

Requirements: UI-001
Task: P2-T008
"""

import readline
import os
from assistant.agent import DealerAssistant
from assistant.retrieval import CatalogueRetriever
from assistant.tools import create_default_tool_registry
import pandas as pd


def main():
    """
    Main entry point for CLI interface.

    Requirements: UI-001
    Task: P2-T008
    """
    # Load catalogue
    catalogue = pd.read_csv("catalogue.csv")

    # Initialize components
    retriever = CatalogueRetriever("catalogue.csv")
    retriever.load_catalogue()
    retriever.initialize_embedding_model()
    retriever.build_vector_store()

    tool_registry = create_default_tool_registry(catalogue)
    assistant = DealerAssistant(retriever, tool_registry)

    print("=" * 60)
    print("VIKMO Dealer Assistant (LLM-Driven)")
    print("=" * 60)

    # Check LLM availability
    llm_status = "✓ LLM available" if assistant.provider.is_available() else "⚠ LLM not available (using fallback)"
    print(f"Status: {llm_status}")
    print("Type 'quit' or 'exit' to end the conversation.")
    print()

    while True:
        try:
            query = input("User: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("Assistant: Goodbye!")
                break

            if not query:
                continue

            response = assistant.process_query(query)
            print(f"Assistant: {response}")
            print()

        except KeyboardInterrupt:
            print("\nAssistant: Goodbye!")
            break
        except Exception as e:
            print(f"Assistant: An error occurred: {e}")
            print()


if __name__ == "__main__":
    main()
