from huggingface_hub import snapshot_download

JOBS = [
    ("sahilmahendrakar/Paradee-8M-v1.0", ["onnx/paradee_int8.onnx", "config.json", "tokenizer.json", "README.md", "LICENSE"]),
    ("Cactus-Compute/whistle", None),
    ("LiquidAI/LFM2.5-2.6B-GGUF", ["LFM2.5-2.6B-Q4_K_M.gguf", "README.md", "LICENSE"]),
    ("ResembleAI/chatterbox", ["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json", "mtl_tokenizer.json", "README.md"]),
    ("ResembleAI/chatterbox-turbo", None),
]

for repo, patterns in JOBS:
    print("DOWNLOAD", repo, flush=True)
    path = snapshot_download(repo_id=repo, allow_patterns=patterns)
    print("DONE", repo, path, flush=True)
print("ALL_DONE", flush=True)
