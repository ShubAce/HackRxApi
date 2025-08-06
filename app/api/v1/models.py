from pydantic import BaseModel, HttpUrl
from typing import List

class QueryRequest(BaseModel):
    # This model now correctly expects a URL for the 'documents' field
    # and a list of questions.
    documents: HttpUrl
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]