"""SHL Assessment Recommender Agent — FAISS retrieval + Groq LLM."""

import json
import re
from groq import Groq
from retriever import CatalogRetriever
from prompts import SYSTEM_PROMPT


class SHLAgent:
    """Handles chat conversations using RAG over the SHL catalog."""

    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.1-8b-instant"
        self.retriever = CatalogRetriever()
        print(f"Agent ready! Using model: {self.model_name}")

    def _build_search_query(self, messages):
        """Combine all user messages into a single search query."""
        user_parts = []
        for msg in messages:
            if msg["role"] == "user":
                user_parts.append(msg["content"])
        return " ".join(user_parts)

    def _get_catalog_context(self, messages):
        """Retrieve relevant assessments for the current conversation."""
        query = self._build_search_query(messages)
        results = self.retriever.search(query, top_k=15)
        return self.retriever.format_for_prompt(results)

    def _format_conversation(self, messages):
        """Format conversation history as text for the prompt."""
        lines = []
        for msg in messages:
            role = msg["role"].upper()
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def _parse_llm_response(self, response_text):
        """Parse the LLM's JSON response, handling markdown fences and extra text."""
        text = response_text.strip()

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _validate_recommendations(self, recommendations):
        """Ensure every recommendation exists in the catalog."""
        valid = []
        catalog_urls = {item["url"] for item in self.retriever.catalog}
        catalog_names = {item["name"].lower(): item for item in self.retriever.catalog}

        for rec in recommendations:
            if not all(k in rec for k in ["name", "url", "test_type"]):
                continue

            if rec["url"] not in catalog_urls:
                matched = catalog_names.get(rec["name"].lower())
                if matched:
                    rec["url"] = matched["url"]
                    rec["test_type"] = matched.get("test_type", rec["test_type"])
                else:
                    continue

            valid.append({
                "name": rec["name"],
                "url": rec["url"],
                "test_type": rec["test_type"]
            })

        return valid[:10]

    def chat(self, messages):
        """Process a chat request and return the agent's response."""
        catalog_context = self._get_catalog_context(messages)
        conversation_text = self._format_conversation(messages)

        full_prompt = SYSTEM_PROMPT.format(
            catalog_context=catalog_context,
            conversation=conversation_text
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": full_prompt},
                    {"role": "user", "content": messages[-1]["content"]}
                ],
                temperature=0.3,
                max_tokens=1500
            )

            llm_output = response.choices[0].message.content
            result = self._parse_llm_response(llm_output)

            if result is None:
                return self._fallback_response()

            recs = result.get("recommendations", [])
            validated_recs = self._validate_recommendations(recs)

            return {
                "reply": result.get("reply", "Could you tell me more about the role?"),
                "recommendations": validated_recs,
                "end_of_conversation": result.get("end_of_conversation", False)
            }

        except Exception as e:
            print(f"Error calling LLM: {e}")
            return self._fallback_response()

    def _fallback_response(self):
        """Safe fallback — always return a valid response."""
        return {
            "reply": "I'd be happy to help you find the right SHL assessments. Could you tell me more about the role you're hiring for and what skills you'd like to evaluate?",
            "recommendations": [],
            "end_of_conversation": False
        }
