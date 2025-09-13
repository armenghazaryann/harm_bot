"""Query entities module.

DEPRECATED: All entities have been replaced by LangChain components.
- Chunk/ChunkType → LangChain Document objects with RecursiveCharacterTextSplitter
- Embedding → LangChain PGVector automatic embeddings
- Utterance → LangChain Document objects with metadata

No imports needed - LangChain handles everything automatically.
"""

# ALL IMPORTS REMOVED - Using LangChain components instead
# from .chunk import Chunk, ChunkType  # → LangChain Document objects
# from .embedding import Embedding     # → LangChain PGVector embeddings
# from .utterance import Utterance     # → LangChain Document objects

__all__ = []  # No entities exported - using LangChain components
