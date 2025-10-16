import os
from dotenv import load_dotenv
load_dotenv("API_KEYs.env")
API_KEY1=os.getenv("OPENAI_API_KEY")
assert API_KEY1, "OPENAI API KEY not found, please, check your API file"

API_KEY2=os.getenv("HF_API_KEY")
assert API_KEY2, "HF API KEY not found, please, check your API file"

