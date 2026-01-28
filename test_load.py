
import torch

print("Trying to load model.pth...")

try:
    sd = torch.load("model.pth", map_location="cpu")
    print("SUCCESS! File is a valid PyTorch model.")
    print("Keys:", list(sd.keys())[:10])
except Exception as e:
    print("FAILED! The file is invalid or corrupted.")
    print("Error:", e)
