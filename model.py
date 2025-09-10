# For pushing a trained model to hugging face : 

from huggingface_hub import notebook_login
notebook_login()

model.push_to_hub("your-username/finetuned-codet5")
tokenizer.push_to_hub("your-username/finetuned-codet5")

# Next time you can load it directly:

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("your-username/finetuned-codet5")
tokenizer = AutoTokenizer.from_pretrained("your-username/finetuned-codet5")

# Export dataset into hugging face

from datasets import Dataset
import huggingface_hub
hf_dataset = Dataset.from_pandas(augmented_xss)

hf_dataset.push_to_hub("your-username/dataset_name")
print("Dataset uploaded to Hugging Face!")


# Import from hugging face


from datasets import load_dataset
import pandas as pd
dataset = load_dataset("your-username/dataset_name", split="train")

augmented_xss = dataset.to_pandas()

# Optional: Display info
print(f"Loaded dataset shape: {augmented_xss.shape}")
print("Label distribution:")
print(augmented_xss['Label'].value_counts())
print("Preview:")
print(augmented_xss.tail())
