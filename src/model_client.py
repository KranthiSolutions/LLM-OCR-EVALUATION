import os
import json

from litellm import completion


def extract_with_llm(
    model_id,
    document_text,
    required_fields
):
    prompt = f"""
    Extract the following fields from the document.

    Required fields:
    {json.dumps(required_fields)}

    Return only a JSON object.
    If a field is not available, return null.
    Do not guess missing information.

    Document:
    {document_text}
    """

    response = completion(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content