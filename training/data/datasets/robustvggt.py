from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="onground/robustvggt",
    repo_type="dataset",
    allow_patterns=[
        "eth3d/rgb/",
        "phototourism/*/dense/images",
        "phototourism/*/dense/sparse",
        "onthego/*",
        "robustnerf/*",
    ],
    local_dir="../robustvggt"
)