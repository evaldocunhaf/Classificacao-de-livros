import requests


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", model: str = "qwen3.5:9b"):
        self.host = host.rstrip("/")
        self.model = model

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        think: bool = True,
        num_predict: int | None = None,
    ) -> str:
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if not think:
            body["think"] = False
        if num_predict is not None:
            body["options"]["num_predict"] = num_predict
        resp = requests.post(
            f"{self.host}/api/generate",
            json=body,
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
            return self.model in [m["name"] for m in r.json().get("models", [])]
        except requests.RequestException:
            return False