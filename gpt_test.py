import os

import requests
import json
import time

# Set your OpenAI API key and daily API quota limit (adjust as needed)
with open("config.json") as f:
    config = json.load(f)
API_KEY = config.get("OPENAI_API_KEY")
API_QUOTA_LIMIT = 100000  # Example daily quota limit in tokens


def check_api_quota():
    """
    Check API usage to monitor remaining quota.

    Returns:
        int: Remaining tokens for the day, or -1 if the quota cannot be checked.
    """
    url = "https://api.openai.com/v1/dashboard/billing/usage"
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        usage_data = response.json()
        remaining_quota = API_QUOTA_LIMIT - usage_data.get("total_usage", 0)
        return remaining_quota
    except requests.exceptions.RequestException as e:
        print(f"Unable to check quota: {e}")
        return -1


def call_gpt4_vision(question: str, image_url: str, max_tokens: int = 100, retries: int = 3) -> str:
    """
    Multimodal call to GPT-4 for image-related questions with retry and quota check.

    Args:
        question (str): The question to ask the model.
        image_url (str): URL of the public image.
        max_tokens (int): Maximum token length for the response.
        retries (int): Number of retry attempts in case of request failure.

    Returns:
        str: The response from GPT-4 or an error message.
    """
    remaining_quota = check_api_quota()
    if remaining_quota != -1 and remaining_quota < max_tokens:
        return "Insufficient quota for this request."

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4-vision",
        "messages": [
            {"role": "user", "content": question},
            {"role": "user", "content": f"[Image URL: {image_url}]"}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            answer = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response")
            return answer
        except requests.exceptions.RequestException as e:
            print(f"Request failed, retry {attempt + 1}/{retries}: {e}")
            time.sleep(2)  # Wait before retrying
    return "Request failed after maximum retries."


def call_gpt3_5(question: str, max_tokens: int = 100, retries: int = 3) -> str:
    """
    Standard call to GPT-3.5 for text-based questions with retry and quota check.

    Args:
        question (str): The question to ask the model.
        max_tokens (int): Maximum token length for the response.
        retries (int): Number of retry attempts in case of request failure.

    Returns:
        str: The response from GPT-3.5 or an error message.
    """
    remaining_quota = check_api_quota()
    if remaining_quota != -1 and remaining_quota < max_tokens:
        return "Insufficient quota for this request."

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": question}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            answer = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response")
            return answer
        except requests.exceptions.RequestException as e:
            print(f"Request failed, retry {attempt + 1}/{retries}: {e}")
            time.sleep(2)  # Wait before retrying
    return "Request failed after maximum retries."
