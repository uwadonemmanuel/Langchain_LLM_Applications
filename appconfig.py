import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

class EnvConfig():

    def __init__(self):
        # self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")

# instantiate environment vars
env_config = EnvConfig()

