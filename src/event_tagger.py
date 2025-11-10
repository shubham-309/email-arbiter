from openai import OpenAI
from typing import List, Dict

client = OpenAI()

EVENT_CLASSES = ["proposal", "approval", "confirmation", "rejection", "request", "other"]

def tag_events(thread_emails: List[Dict]) -> List[Dict[str, str]]:
    """Tag each email in thread with event type."""
    tags = []
    for email in thread_emails:
        prompt = f"""
        Classify this email body into one event: {', '.join(EVENT_CLASSES)}.
        Body: {email['body_clean']}
        Output: JSON {"event": "class"}
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        tag = json.loads(response.choices[0].message.content)
        tags.append({"id": email['id'], **tag})
    return tags