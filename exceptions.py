# exceptions.py

class RAGException(Exception):
    """
    Base class for all RAG application errors.
    Never raise this directly — raise a subclass.
    Catching this catches ALL RAG errors in one place if needed.
    """
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(message)
        self.message = message
        self.original_error = original_error
    
    def __str__(self):
        if self.original_error:
            return f"{self.message} | Caused by: {type(self.original_error).__name__}: {self.original_error}"
        return self.message


class RetrievalError(RAGException):
    """
    Vector store search failed.
    Could be: ChromaDB not loaded, index corrupted, embedding failed.
    HTTP status: 500
    """
    pass


class DocumentNotFoundError(RAGException):
    """
    Search returned no results.
    Not a crash — the system worked, just found nothing relevant.
    HTTP status: 404
    """
    pass


class LLMError(RAGException):
    """
    Language model call failed.
    Could be: rate limit, timeout, invalid response, model unavailable.
    HTTP status: 500 or 429 depending on cause.
    """
    pass


class IngestionError(RAGException):
    """
    Document ingestion failed.
    Could be: file unreadable, chunking failed, embedding failed,
    ChromaDB write failed.
    HTTP status: 500
    """
    pass


class VectorStoreNotInitializedError(RAGException):
    """
    app.state.vectorstore doesn't exist yet.
    Means the lifespan startup failed or hasn't completed.
    HTTP status: 503 (Service Unavailable — not your fault, try again)
    """
    pass