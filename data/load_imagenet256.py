import kagglehub

# Download latest version
path = kagglehub.dataset_download("dimensi0n/imagenet-256")

print("Path to dataset files:", path)