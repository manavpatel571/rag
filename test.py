import os
import base64
from huggingface_hub import InferenceClient

# Initialize client
client = InferenceClient(
    provider="auto",
    api_key=os.environ["HF_TOKEN"],
)

# Path to your local image
image_path = r"D:\old_laptop\startup\all_task\nlp_rag\extracted_images\1706.03762v7_page_3_img_1.png"  # <-- change this

# Read and encode image as base64
with open(image_path, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

# Run inference
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one sentence."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            ],
        }
    ],
)

# ✅ Fix: print message directly
print(response.choices[0].message)
