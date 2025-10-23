import os

from dotenv import load_dotenv
from huggingface_hub import login


package_dir = os.path.dirname(__file__)
package_dir = os.path.dirname(package_dir)
package_dir = os.path.dirname(package_dir)
env_path = os.path.join(package_dir, ".env")

print("Env path:", env_path)
load_dotenv(env_path)
login(token=os.environ["HF_TOKEN"])
