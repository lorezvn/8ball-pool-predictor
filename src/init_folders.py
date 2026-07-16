import os

def create_folders():
    if not os.path.exists("models"): 
        os.makedirs("models")