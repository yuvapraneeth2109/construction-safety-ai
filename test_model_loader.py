from src.model_loader import load_models

models = load_models()

print("Loaded models:")
for name in models:
    print("-", name)
