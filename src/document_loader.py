import fitz


def extract_pdf_text(file_path):
    text_parts = []

    with fitz.open(file_path) as document:
        for page in document:
            text_parts.append(page.get_text("text"))

    return "\n".join(text_parts)