from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct")
