import httpx
from dotenv import dotenv_values

cfg = dotenv_values(".env")
key = cfg.get("OPENAI_API_KEY")
base_url = cfg.get("OPENAI_BASE_URL")
model = cfg.get("AI_MODEL")

headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json"
}
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "hello"}]
}

try:
    resp = httpx.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=10.0)
    print("Status:", resp.status_code)
    print("Body:", resp.text)
except Exception as e:
    print("Error:", e)

