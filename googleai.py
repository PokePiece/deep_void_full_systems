from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()


response = client.models.generate_content(
    model="gemini-2.5-flash", contents="request/token limits for google flash 2.5 api on free tier"
)
print(response.text)