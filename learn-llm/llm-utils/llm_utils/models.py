import requests


def invoke_llama2(prompt: str) -> str:
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama2", "prompt": prompt, "stream": False},
    )
    assert resp.status_code == 200
    return resp.json()["response"]
