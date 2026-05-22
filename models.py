from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


class QueryRequest(BaseModel):
    """
    What the user sends when asking a question.
    Pydantic validates this automatically — if the data doesn't
    match, it raises a clear error BEFORE your code runs.
    """
    question: str = Field(
        min_length=3,
        max_length=500,
        description="The question to ask the knowledge base"
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of chunks to retrieve"
    )
    
    @field_validator("question")
    def question_must_not_be_blank(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be blank or only whitespace")
        return v.strip()


class SourceDocument(BaseModel):
    """A single retrieved chunk with its metadata."""
    content: str
    source: str = "unknown"
    chunk_index: Optional[int] = None


class QueryResponse(BaseModel):
    """What your app sends back."""
    answer: str
    sources: list[SourceDocument]
    question: str
    retrieval_count: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConfluenceIngestRequest(BaseModel):
    """
    Credentials and config for ingesting a Confluence space.
    Supports both real Atlassian Cloud instances and mock mode.
    """
    base_url: str = Field(
        description="Confluence base URL, e.g. https://yourorg.atlassian.net"
    )
    username: str = Field(
        description="Atlassian account email"
    )
    api_token: str = Field(
        description="Atlassian API token (not your password)"
    )
    space_key: str = Field(
        description="Confluence space key, e.g. ENG. Use 'MOCK' for built-in demo data."
    )
    max_pages: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum pages to fetch from the space"
    )
    mock: bool = Field(
        default=False,
        description="If True, uses built-in mock data — no real Confluence needed"
    )

    @field_validator("base_url")
    def base_url_must_start_with_http(cls, v):
        if not v.startswith("http"):
            raise ValueError("base_url must start with http:// or https://")
        return v.rstrip("/")

    @field_validator("space_key")
    def space_key_uppercase(cls, v):
        return v.strip().upper()


class JiraIngestRequest(BaseModel):
    """
    Credentials and config for ingesting a JIRA project.
    Supports both real Atlassian Cloud instances and mock mode.
    """
    base_url: str = Field(
        description="JIRA base URL, e.g. https://yourorg.atlassian.net"
    )
    username: str = Field(
        description="Atlassian account email"
    )
    api_token: str = Field(
        description="Atlassian API token"
    )
    project_key: str = Field(
        description="JIRA project key, e.g. QA. Use 'MOCK' for built-in demo data."
    )
    max_issues: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum issues to fetch from the project"
    )
    mock: bool = Field(
        default=False,
        description="If True, uses built-in mock data — no real JIRA needed"
    )

    @field_validator("base_url")
    def base_url_must_start_with_http(cls, v):
        if not v.startswith("http"):
            raise ValueError("base_url must start with http:// or https://")
        return v.rstrip("/")

    @field_validator("project_key")
    def project_key_uppercase(cls, v):
        return v.strip().upper()
