from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=500,
        description="The question to ask the knowledge base",
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of chunks to retrieve",
    )

    @field_validator("question")
    def question_must_not_be_blank(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be blank or only whitespace")
        return v.strip()


class SourceDocument(BaseModel):
    content: str
    source: str = "unknown"
    chunk_index: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    question: str
    retrieval_count: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConfluenceIngestRequest(BaseModel):
    """
    Credentials and config for ingesting a Confluence space.

    base_url: root URL of your Confluence instance including the context path.
        e.g. https://yourorg.atlassian.net/wiki   (Atlassian Cloud)
             https://yourorg.example.com/confluence  (self-hosted Server)

    You may also paste any full page URL — the context path is extracted
    automatically, so /spaces/... and /pages/... are stripped.

    space_key: the space key (e.g. ENG) or any full page URL —
        the key is extracted automatically.

    mock: if True, uses built-in demo data. No real Confluence needed.
    """
    base_url:  str = Field(description="Confluence instance base URL including context path")
    username:  str = Field(description="Confluence username or email address")
    api_token: str = Field(description="API token (Cloud) or password (Server)")
    space_key: str = Field(description="Space key or any page URL from the space")
    max_pages: int = Field(default=50, ge=1, le=500)
    mock:      bool = Field(default=False)

    @field_validator("base_url")
    def base_url_not_empty(cls, v):
        if not v.strip():
            raise ValueError("base_url is required")
        return v.strip().rstrip("/")

    @field_validator("space_key")
    def space_key_not_empty(cls, v):
        if not v.strip():
            raise ValueError("space_key is required")
        return v.strip()


class JiraIngestRequest(BaseModel):
    """
    Credentials and config for ingesting a JIRA project.

    base_url: root URL of your JIRA instance.
        e.g. https://yourorg.atlassian.net   (Atlassian Cloud)
             https://jira.yourorg.com         (self-hosted)

    project_key: JIRA project key (e.g. QA, ENG) or any issue URL —
        the key is extracted automatically.

    mock: if True, uses built-in demo data. No real JIRA needed.
    """
    base_url:    str = Field(description="JIRA instance base URL")
    username:    str = Field(description="JIRA username or email address")
    api_token:   str = Field(description="API token (Cloud) or password (Server)")
    project_key: str = Field(description="Project key or any issue URL from the project")
    max_issues:  int = Field(default=100, ge=1, le=1000)
    mock:        bool = Field(default=False)

    @field_validator("base_url")
    def base_url_not_empty(cls, v):
        if not v.strip():
            raise ValueError("base_url is required")
        return v.strip().rstrip("/")

    @field_validator("project_key")
    def project_key_not_empty(cls, v):
        if not v.strip():
            raise ValueError("project_key is required")
        return v.strip().upper()
