import httpx
from dotenv import dotenv_values

cfg = dotenv_values(".env")
key = cfg.get("OPENAI_API_KEY")
base_url = cfg.get("OPENAI_BASE_URL")

models = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-flash-latest"
]

for m in models:
    try:
        r = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": m, "messages": [{"role": "user", "content": "hi"}]},
            timeout=8.0
        )
        print(f"Model: {m} -> Status: {r.status_code}")
        if r.status_code != 200:
            print("   Error:", r.text[:120])
    except Exception as e:
        print(f"Model: {m} -> Failed: {e}")

