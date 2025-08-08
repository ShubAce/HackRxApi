import httpx
import tempfile
import hashlib
from pypdf import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.utils.logger import get_logger
from typing import List, Dict, Any

logger = get_logger(__name__)

class DocumentService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )

    async def get_document_from_url(self, url: str) -> tuple[bytes, str]:
        """Asynchronously downloads a document from a URL."""
        logger.info(f"Downloading document from: {url}")
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, follow_redirects=True, timeout=30.0)
                response.raise_for_status()
                content = response.content
                file_extension = url.split('?')[0].split('.')[-1].lower()
                return content, file_extension
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error downloading document: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to download document: {e}")
                raise

    def _parse_pdf(self, content: bytes) -> str:
        """Extracts text from PDF content."""
        logger.info("Parsing PDF document.")
        text = ""
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
            try:
                reader = PdfReader(tmp_path)
                for page in reader.pages:
                    text += page.extract_text() or ""
            except Exception as e:
                logger.error(f"Error reading PDF file: {e}")
                raise
        return text

    def parse_document(self, content: bytes, file_type: str) -> str:
        """Parses document based on file type."""
        if file_type == 'pdf':
            return self._parse_pdf(content)
        else:
            logger.warning(f"Unsupported file type: {file_type}. Skipping parsing.")
            raise ValueError(f"Unsupported file type: {file_type}")

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Splits a large text into smaller chunks."""
        logger.info("Chunking document text.")
        docs = self.text_splitter.create_documents([text])
        chunks = [{"text": doc.page_content, "metadata": doc.metadata} for doc in docs]
        logger.info(f"Document split into {len(chunks)} chunks.")
        return chunks

    def generate_document_namespace(self, url: str) -> str:
        """Creates a unique and deterministic namespace from the document URL."""
        return hashlib.sha256(url.encode()).hexdigest()

document_service = DocumentService()
