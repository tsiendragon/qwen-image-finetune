import os

from dotenv import load_dotenv
from huggingface_hub import login


package_dir = os.path.dirname(__file__)
package_dir = os.path.dirname(package_dir)
package_dir = os.path.dirname(package_dir)
env_path = os.path.join(package_dir, ".env")

if not os.environ.get("QFLUX_DOTENV_LOADED"):
    load_dotenv(env_path)
    os.environ["QFLUX_DOTENV_LOADED"] = "1"
    print("Environment variables loaded from .env file")
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("Logged in to Hugging Face")
    else:
        print("HF_TOKEN not set, skipping Hugging Face login (offline mode)")
