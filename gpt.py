import glob
import os
import random
import re

import requests
import json
import time
from collections import OrderedDict

with open("api.json") as f:
    config = json.load(f)
API_KEY = config.get("OPENAI_API_KEY")
API_QUOTA_LIMIT = 100000  # Example daily quota limit in tokens

prompt_file = "prompt.txt"

PROXY = {
    "http": "http://127.0.0.1:8080",
    "https": "http://127.0.0.1:8080"
}
with open(prompt_file, 'r') as f:
    prompt = f.read()


# def check_api_quota() -> int:
#     """
#     Check API usage to monitor the remaining quota, supporting proxy settings.
#
#     Args:
#         proxy (dict): Proxy settings as a dictionary (e.g., {"http": "http://proxyserver:port", "https": "http://proxyserver:port"}).
#
#     Returns:
#         int: Remaining tokens for the day, or -1 if the quota cannot be checked.
#     """
#     url = "https://api.openai.com/v1/dashboard/billing/usage"
#     headers = {
#         "Authorization": f"Bearer {API_KEY}"
#     }
#     try:
#         # ????????
#         response = requests.get(url, headers=headers, proxies=PROXY)
#         response.raise_for_status()
#         usage_data = response.json()
#         remaining_quota = API_QUOTA_LIMIT - usage_data.get("total_usage", 0)
#         return remaining_quota
#     except requests.exceptions.RequestException as e:
#         print(f"Unable to check quota: {e}")
#         return -1


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
    # remaining_quota = check_api_quota()
    # if remaining_quota != -1 and remaining_quota < max_tokens:
    #     return "Insufficient quota for this request."

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


def get_last_frame_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    frame_keys = [key for key in data.keys() if key.isdigit()]

    last_frame_key = str(max(map(int, frame_keys)))

    return data[last_frame_key]


def get_frame_data(json_path, frame_number=-1):
    with open(json_path, 'r') as f:
        data = json.load(f)

    frame_keys = [key for key in data.keys() if key.isdigit()]

    frame_keys = sorted(map(int, frame_keys))

    max_frame_key = frame_keys[-1]

    if frame_number == -1:
        random_frame_key = str(random.choice(frame_keys))
        return data[random_frame_key]
    elif frame_number in frame_keys:
        return data[str(frame_number)]
    else:
        return data[str(max_frame_key)]


def call_gpt(question: str, model_version: str = "gpt-3.5-turbo", max_tokens: int = 100, retries: int = 3) -> str:
    """
    Call GPT model (3.5, 4, 4-turbo) with retry and quota check, supporting proxy settings.

    Args:
        question (str): The question to ask the model.
        model_version (str): The model version to use ("gpt-3.5-turbo", "gpt-4", "gpt-4-turbo").
        max_tokens (int): Maximum token length for the response.
        retries (int): Number of retry attempts in case of request failure.
        proxy (dict): Proxy settings as a dictionary (e.g., {"http": "http://proxyserver:port", "https": "http://proxyserver:port"}).

    Returns:
        str: The response from the specified GPT model or an error message.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_version,
        "messages": [
            {"role": "user", "content": question}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), proxies=PROXY)
            response.raise_for_status()
            response_data = response.json()

            answer = response_data.get("choices", [{}])[0].get("message", {}).get("content", "No response")

            usage_info = response_data.get("usage", {})
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
            total_tokens = usage_info.get("total_tokens", 0)
            print(f"Tokens used: Prompt = {prompt_tokens}, Completion = {completion_tokens}, Total = {total_tokens}")

            return answer
        except requests.exceptions.RequestException as e:
            print(f"Request failed, retry {attempt + 1}/{retries}: {e}")
            time.sleep(2)  # Wait before retrying
    return "Request failed after maximum retries."


def extract_json(response):
    try:
        json_text = re.search(r'{.*}', response, re.DOTALL).group(0)
        result = json.loads(json_text)
        return result
    except (json.JSONDecodeError, AttributeError):
        print("Failed to extract JSON from the response.")
        return None


def add_answer1_to_database(response_json, database, max_size=30):
    """
    Adds the answer1 from response_json to the database and ensures the database size
    does not exceed max_size. If the database exceeds max_size, it removes the oldest
    entry before adding the new one.

    Parameters:
    - response_json: JSON data containing answer1
    - database: OrderedDict storing answer1 data
    - max_size: Maximum size of the database; removes the oldest entry if exceeded

    Returns:
    - Updated database
    """
    if "answer1" in response_json:
        # Ensure database is an OrderedDict
        if not isinstance(database, OrderedDict):
            database = OrderedDict(database)

        # If the database is full, remove the oldest entry
        if len(database) >= max_size:
            database.popitem(last=False)  # Removes the first-added item in OrderedDict

        # Add the new entry
        new_key = str(int(max(database.keys(), key=int)) + 1) if database else "0"
        database[new_key] = response_json["answer1"]
        return database
    else:
        print("answer1 not found in response JSON.")
        return database


def get_overall_similarity(response_json):
    if "answer2" in response_json and "Overall Similarity" in response_json["answer2"]:
        try:
            return int(float(response_json["answer2"]["Overall Similarity"]))
        except ValueError:
            print("Failed to convert Overall Similarity to integer.")
            return None
    else:
        print("Overall Similarity not found in answer2.")
        return None


def modify_json_file(json_file, mutate_info):
    # Load JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    # Extract vehicle information from mutate_info
    vehicle_id = mutate_info["Vehicle ID"]
    new_location = mutate_info["Location"]
    new_speed = mutate_info["Speed"]

    # Recursively search and modify NPCs
    for key, value in data.items():
        if isinstance(value, dict) and "NPC" in value:
            for npc in value["NPC"]:
                if npc["id"] == vehicle_id:
                    # Modify velocity
                    velocity = npc.get("velocity", {})
                    current_speed = (velocity.get("x", 0) ** 2 + velocity.get("y", 0) ** 2 + velocity.get("z",
                                                                                                          0) ** 2) ** 0.5
                    if current_speed != 0:
                        scale = new_speed / current_speed
                        npc["velocity"]["x"] *= scale
                        npc["velocity"]["y"] *= scale
                        npc["velocity"]["z"] *= scale
                    else:
                        # If current speed is zero, set velocity based on new speed equally divided among components
                        direction = [1, 0, 0]  # Default direction if current speed is zero
                        npc["velocity"]["x"] = new_speed * direction[0]
                        npc["velocity"]["y"] = new_speed * direction[1]
                        npc["velocity"]["z"] = new_speed * direction[2]

                    # Modify location (keeping rotation unchanged)
                    if "transform" in npc and "location" in npc["transform"]:
                        npc["transform"]["location"]["x"] = new_location["x"]
                        npc["transform"]["location"]["y"] = new_location["y"]
                    break
    return data

def get_answer3_vehicle_info(response_json):
    if "answer3" in response_json:
        try:
            vehicle_info = {
                "Vehicle ID": response_json["answer3"]["Modified Background Vehicle for diversity"]["Vehicle ID"],
                "Location": response_json["answer3"]["Modified Background Vehicle for diversity"]["Location"],
                "Speed": response_json["answer3"]["Modified Background Vehicle for diversity"]["Speed"]
            }
            return vehicle_info
        except KeyError as e:
            print(f"Missing key in answer3: {e}")
            return None
    else:
        print("answer3 not found in response JSON.")
        return None


def get_all_json_files(base_dir):
    json_files = []
    for root, dirs, files in os.walk(base_dir):
        dirs.sort()
        files.sort()
        for filename in files:
            if filename.endswith('.json'):
                json_files.append(os.path.join(root, filename))
    return json_files


if __name__ == "__main__":
    Scenario_database = {}
    json_files = get_all_json_files('./json_out')
    if not os.path.exists('./new_json_files'):
        os.mkdir('./new_json_files')
    for json_file in json_files:
        print(f"Processing {json_file}...")
        try:
            # gpt
            scenario_description = str(get_frame_data(json_file)).replace("\n", "").replace(' ', '')
            question = prompt + "\n scenario snapshot:\n" + str(
                scenario_description + "\n___\n Scenario-dict\n" + str(Scenario_database))
            response = call_gpt(question, model_version="gpt-4-turbo", max_tokens=1500)
            print("Response:", response)
            response_json = extract_json(response)
            Scenario_database = add_answer1_to_database(response_json, Scenario_database, 30)
            answer3_vehicle_info = get_answer3_vehicle_info(response_json)
            print("Answer3 Vehicle Info:", answer3_vehicle_info)
            mutate_info = answer3_vehicle_info
            data = modify_json_file(json_file, mutate_info)
            # Save modified JSON file
            new_json_file = json_file.replace('.json', '_modified.json')
            with open(new_json_file, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Modified JSON file saved as: {new_json_file}")
        # reload scenario state
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

