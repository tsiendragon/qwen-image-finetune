import os
from huggingface_hub import login

login(token=os.environ["HF_TOKEN"])
