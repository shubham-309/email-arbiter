import pytest
from src.data_layer import load_emails

def test_load():
    df = load_emails("data/raw/emails.jsonl")
    assert len(df) == 100  # From dataset
    assert pd.api.types.is_datetime64_any_dtype(df['date'])