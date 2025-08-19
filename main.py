# Main entry point
from src.train import train_model

if __name__ == "__main__":
    model = train_model()
    print("Model trained and ready!")
