# test_api.py
import requests
import os
from dotenv import load_dotenv

load_dotenv('.env')

print("🔍 Testing OpenRouter API...\n")

api_key = os.getenv('OPENROUTER_API_KEY')
model = os.getenv('OPENROUTER_MODEL')

print(f"API Key: {api_key[:20]}...{api_key[-10:]}")
print(f"Model: {model}\n")

# Test API call
response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/reflexion",
    },
    json={
        "model": model,
        "messages": [{"role": "user", "content": "Say 'API working!' if you can read this."}],
        "max_tokens": 50
    }
)

print(f"Status Code: {response.status_code}")

if response.status_code == 200:
    result = response.json()
    print(f"✅ SUCCESS!")
    print(f"Response: {result['choices'][0]['message']['content']}")
    print(f"\n💰 Model used: {result.get('model', 'N/A')}")
else:
    print(f"❌ ERROR!")
    print(f"Response: {response.text}")
