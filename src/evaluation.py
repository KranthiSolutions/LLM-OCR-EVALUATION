import json
import re

def normalize(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("–", "-").replace("—", "-")
    return text

def flatten_json(obj, prefix=""):
    """Flatten nested JSON into dotted paths; lists remain canonical JSON values."""
    out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                out.update(flatten_json(value, path))
            else:
                out[path] = value
    else:
        out[prefix or "value"] = obj
    return out

def compare_values(expected: dict, predicted: dict):
    """Compare expected fields with predicted fields; missing predictions count as wrong."""
    comparisons = []
    for key, expected_value in expected.items():
        predicted_value = predicted.get(key, "")
        comparisons.append({
            "field": key,
            "expected": expected_value if not isinstance(expected_value, (dict, list)) else json.dumps(expected_value, ensure_ascii=False),
            "predicted": predicted_value if not isinstance(predicted_value, (dict, list)) else json.dumps(predicted_value, ensure_ascii=False),
            "match": normalize(expected_value) == normalize(predicted_value),
        })
    return comparisons
