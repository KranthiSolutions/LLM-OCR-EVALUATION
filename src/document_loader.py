from io import BytesIO
import fitz
from PIL import Image

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract embedded text from a PDF. Scanned PDFs may return little or no text."""
    chunks = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                chunks.append(f"--- PAGE {page_num} ---\n{text}")
    return "\n\n".join(chunks)

def pdf_to_images(pdf_bytes: bytes, max_pages: int = 5):
    """Render PDF pages to PNG bytes for vision models when no embedded text is found."""
    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page in list(pdf)[:max_pages]:
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            images.append(pix.tobytes("png"))
    return images
