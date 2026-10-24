import os
import sys
from dotenv import load_dotenv
from sarvamai import SarvamAI

# Load variables from .env file (check script dir and current dir)
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=env_path)
load_dotenv()

api_key = os.getenv("SARVAM_API_KEY")
if not api_key:
    print("Error: SARVAM_API_KEY not found in environment. Please check your .env file.")
    sys.exit(1)

# Initialize Sarvam client (reads SARVAM_API_KEY from environment)
client = SarvamAI(api_subscription_key=api_key)

# Make chat completion call with sarvam-105b-conversations
response = client.chat.completions(
    model="sarvam-105b-conversations",
    messages=[
        {"role": "user", "content": "Hello, introduce yourself as Disha in one short sentence."}
    ],
)

# Print the model reply
print(response.choices[0].message.content)
