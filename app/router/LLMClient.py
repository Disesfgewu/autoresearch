import dotenv
import os
import requests

from pydantic import BaseModel

dotenv.load_dotenv()

llm_router_url = os.getenv("LLM_ROUTER_URL")
llm_router_api_key = os.getenv("LLM_ROUTER_API_KEY")

class LLMClient:
    def __init__(self, logger, url=llm_router_url, api_key=llm_router_api_key):
        self.url = url
        self.api_key = api_key
        self.classifier = "gemma-3-27b-it"
        self.logger = logger

    # ======= General LLM Query and get response =======
    def ask(self, prompt):
        """
        input:
            prompt: str - The prompt to send to the LLM
        output:
            response: dict - The response from the LLM
        """
        self.logger.info(f"LLMClient.ask() prompt: {prompt}")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            url=f"{self.url}/v1/completions",
            headers=headers,
            json={"prompt": prompt},
            timeout=90,
        )
        response.raise_for_status()
        self.logger.info(f"LLMClient.ask() response: {response.text}")
        return response.json()
    
    def discuss(self, messages):
        """
        input:
            messages: list - The messages to send to the LLM
        output:
            response: dict - The response from the LLM
        """
        self.logger.info(f"LLMClient.discuss() messages: {messages}")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            url=f"{self.url}/v1/chat/completions",
            headers=headers,
            json={"messages": messages},
            timeout=90,
        )

        response.raise_for_status()
        self.logger.info(f"LLMClient.discuss() response: {response.text}")
        return response.json()
    
    # ======= Classifier for Intent Recognition =======
    def intentTemplate(self, userInput, categories):
        """
        input:
            userInput: str - The user input to classify
            categories: list[str] - The categories to classify into
        output:
            category: str - The category that the user input belongs to
        """
        class Intent(BaseModel):
            category: str

        prompt = f"Classify the following user input into one of the following categories: {categories}."
        prompt += f"\n\nUser Input: {userInput}"
        prompt += f"\n\nRespond ONLY with a JSON object in the following format: {{\"category\": \"the category that the user input belongs to\"}}"

        response = self._direct_query(prompt)
        if "error" in response:
            raise RuntimeError(f"LLM error: {response['error']['message']}")
        text = response["choices"][0]["message"]["content"].strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        intent = Intent.model_validate_json(text)
        return intent.category

    def _direct_query(self, prompt):
        """
        input:
            prompt: str - The prompt to send to the LLM
        output:
            response: dict - The response from the LLM
        """
        self.logger.info(f"LLMClient._direct_query() prompt: {prompt}")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            url=f"{self.url}/v1/direct_query",
            headers=headers,
            json={
                "model_name": self.classifier,
                "provider": "Google",
                "prompt": prompt,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 20
            },
            timeout=90,
        )
        response.raise_for_status()
        self.logger.info(f"LLMClient._direct_query() response: {response.text}")
        return response.json()

    # ======= Embedding Generation for RAG =======
    def embedding(self, text):
        """
        input:
            text: str or list[str] - The text(s) to generate embedding for
        output:
            embedding: list[float] - 3072-dim embedding vector of the first input
        """
        self.logger.info(f"LLMClient.embedding() text: {text}")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            url=f"{self.url}/v1/embeddings",
            headers=headers,
            json={
                "model": "gemini-embedding-2-preview",
                "input": text
            },
            timeout=90,
        )
        response.raise_for_status()
        self.logger.info(f"LLMClient.embedding() response: {response.text}")
        data = response.json()
        return data["data"][0]["embedding"]
