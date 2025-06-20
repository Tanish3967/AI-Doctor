import kagglehub

# Download latest version
path = kagglehub.dataset_download("kmader/siim-medical-images")

print("Path to dataset files:", path)
