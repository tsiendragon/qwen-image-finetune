import os

from dotenv import load_dotenv
from huggingface_hub import HfFolder, login


package_dir = os.path.dirname(__file__)
package_dir = os.path.dirname(package_dir)
package_dir = os.path.dirname(package_dir)
env_path = os.path.join(package_dir, ".env")
load_dotenv(env_path)
print("Environment variables loaded from .env file")

# Only login if not already logged in
if "HF_TOKEN" not in os.environ or not HfFolder.get_token():
    login(token=os.environ["HF_TOKEN"])
    print("Logged in to Hugging Face")
