import pandas as pd


def normalize_value(value):
    if value is None:
        return ""

    return str(value).strip().casefold()


def compare_fields(expected, extracted):
    results = []

    for field, expected_value in expected.items():
        extracted_value = extracted.get(field)

        expected_normalized = normalize_value(
            expected_value
        )

        extracted_normalized = normalize_value(
            extracted_value
        )

        correct = (
            expected_normalized
            == extracted_normalized
        )

        results.append({
            "field": field,
            "expected": expected_value,
            "extracted": extracted_value,
            "correct": correct
        })

    return pd.DataFrame(results)