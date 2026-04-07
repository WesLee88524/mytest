import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
    local_dir=os.path.expanduser("~/models/Qwen2.5-VL-7B-Instruct"),
    resume_download=True,
)
print("下载完成")
