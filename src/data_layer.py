import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

def load_emails(file_path: str) -> pd.DataFrame:
    """Load JSONL and normalize (parse dates, clean bodies)."""
    emails = []
    with open(file_path, 'r') as f:
        for line in f:
            email = json.loads(line.strip())
            # Normalize date
            email['date'] = pd.to_datetime(email['date'])
            # Clean body: strip quotes, extra newlines
            email['body_clean'] = email['body'].replace('>', '').strip().replace('\n\n', '\n')
            emails.append(email)
    df = pd.DataFrame(emails)
    df = df.sort_values('date').reset_index(drop=True)
    return df

def normalize_to_dict(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert DF to list of dicts for state."""
    return df.to_dict('records')