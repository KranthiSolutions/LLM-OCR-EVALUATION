import base64
import json
import re
from litellm import completion

SYSTEM_PROMPT = """You extract structured fields from synthetic healthcare documents for software testing.
Return valid JSON only. Do not add markdown fences or commentary. Do not infer missing values.
Use the document's own date/time values; use ISO YYYY-MM-DD dates when unambiguous.
Preserve lists and table rows. If a field is absent, omit it or use null.
This is not a clinical decision task."""

def _image_content(image_bytes: bytes):
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}

def _extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model response did not contain a JSON object.")
    return json.loads(text[start:end + 1])

def run_extraction(model_id, document_text, document_type, page_images=None,
                   temperature=0.0, max_tokens=1800):
    user_text = (
        f"Document type: {document_type}\n"
        "Extract the document into a JSON object with a top-level key named 'fields'. "
        "The value of 'fields' must contain the extracted fields and nested appointment/table rows. "
        "Use the same semantic structure as the source document.\n\n"
        f"Document text:\n{document_text[:45000] if document_text else '[No embedded text; inspect page images.]'}"
    )
    content = [{"type": "text", "text": user_text}]
    if page_images:
        content.extend(_image_content(img) for img in page_images[:5])
    response = completion(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    message = response.choices[0].message.content
    parsed = _extract_json(message if isinstance(message, str) else json.dumps(message))
    # Accept both {"fields": {...}} and direct object responses.
    if isinstance(parsed, dict) and isinstance(parsed.get("fields"), dict):
        return parsed["fields"]
    return parsed
