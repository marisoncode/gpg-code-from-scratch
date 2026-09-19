import asyncio
import json
from dotenv import dotenv_values
from openai import AsyncOpenAI

cfg = dotenv_values(".env")
client = AsyncOpenAI(
    api_key=cfg.get("OPENAI_API_KEY"),
    base_url=cfg.get("OPENAI_BASE_URL")
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_batch_by_id",
            "description": "Fetch batch details by batch number or lot number",
            "parameters": {
                "type": "object",
                "properties": {
                    "batch_id": {"type": "string", "description": "The batch lot number"}
                },
                "required": ["batch_id"]
            }
        }
    }
]

async def test():
    messages = [{"role": "user", "content": "What is the status of batch 2026262000?"}]
    print("Calling pass 1...", flush=True)
    resp1 = await client.chat.completions.create(
        model="gemini-flash-latest",
        messages=messages,
        tools=tools
    )
    msg1 = resp1.choices[0].message
    print("Choice 1 content:", msg1.content, flush=True)
    print("Tool calls:", msg1.tool_calls, flush=True)
    
    # Simulate tool return
    if hasattr(msg1, "model_dump"):
        dumped = msg1.model_dump()
    elif hasattr(msg1, "to_dict"):
        dumped = msg1.to_dict()
    else:
        dumped = {"role": "assistant", "content": msg1.content, "tool_calls": [tc.model_dump() for tc in msg1.tool_calls]}
    
    # Clean None fields
    assistant_payload = {
        "role": "assistant",
        "content": dumped.get("content"),
        "tool_calls": dumped.get("tool_calls")
    }
    messages.append(assistant_payload)
    messages.append({
        "role": "tool",
        "tool_call_id": msg1.tool_calls[0].id,
        "content": json.dumps({"batch": {"batch_number": "2026262000", "product": "ben_test_5999", "status": "Production"}})
    })
    
    print("Calling pass 2 with tool result...", flush=True)
    resp2 = await client.chat.completions.create(
        model="gemini-flash-latest",
        messages=messages
    )
    msg2 = resp2.choices[0].message
    print("Choice 2 content:\n", msg2.content, flush=True)

asyncio.run(test())

