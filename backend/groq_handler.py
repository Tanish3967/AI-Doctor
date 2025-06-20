from groq import Groq
import json

class GroqHandler:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)

    def get_response(self, prompt, mode):
        system_prompt = self._get_system_prompt(mode)
        response = self.client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        return self._parse_response(response.choices[0].message.content, mode)

    def _get_system_prompt(self, mode):
        prompts = {
            'emotional': "You are a compassionate emotional support assistant...",
            'medical': "You are a medical diagnosis assistant...",
        }
        return prompts.get(mode, prompts['emotional'])

    def _parse_response(self, response, mode):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"content": response, "confidence": 0.8}
