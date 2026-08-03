import os

def create_folders():
    os.makedirs("models", exist_ok=True)
    os.makedirs("output", exist_ok=True)