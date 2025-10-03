import requests
from dots_ocr.utils.image_utils import PILimage_to_base64
import os
import json

def inference_with_vllm(
        image,
        prompt, 
        ip="localhost",
        port=8000,
        temperature=0.1,
        top_p=0.9,
        max_completion_tokens=32768,
        model_name='model',
        timeout_seconds=60,
        ):
    
    base = f"http://{ip}:{port}/v1"
    url = f"{base}/chat/completions"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": PILimage_to_base64(image)},
                },
                {"type": "text", "text": f"<|img|><|imgpad|><|endofimg|>{prompt}"}
            ],
        }
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_completion_tokens": int(max_completion_tokens) if max_completion_tokens is not None else None
    }

    payload = {k: v for k, v in payload.items() if v is not None}

    api_key = os.environ.get("API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if api_key != "":
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
        resp.raise_for_status()
        data = resp.json()
        # expected OpenAI-compatible shape: choices[0].message.content
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            # fallback: return whole json if shape unexpected
            return data
    except requests.exceptions.RequestException as e:
        print(f"request error: {e}")
        return None
