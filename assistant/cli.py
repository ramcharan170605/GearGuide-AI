"""
CLI interface for Dealer Assistant

Provides interactive command-line interface for the assistant.

Requirements: UI-001
Task: P2-T008
"""

import readline
from assistant.agent import DealerAssistant
from assistant.retrieval import CatalogueRetriever
from assistant.tools import ToolRegistry


def main():
    """
    Main entry point for CLI interface.
    
    Requirements: UI-001
    Task: P2-T008
    """
    # Initialize components
    catalogue_path = "catalogue.csv"
    retriever = CatalogueRetriever(catalogue_path)
    retriever.load_catalogue()
    retriever.initialize_embedding_model()
    retriever.build_vector_store()
    
    tool_registry = ToolRegistry()
    # Register tools here
    
    assistant = DealerAssistant(retriever, tool_registry)
    
    print("=" * 60)
    print("VIKMO Dealer Assistant")
    print("=" * 60)
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
