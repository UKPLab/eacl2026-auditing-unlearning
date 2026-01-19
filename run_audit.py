import argparse
import os
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Import our RINE estimator
from src.rine import RINE

# 1. Data Loading Logic (Adapted from your script)

def load_and_preprocess_dataset(dataset_name, forget_subset, retain_subset, seed=42, test_size=0.1):
    """
    Loads TOFU dataset, combines forget/retain, and assigns binary labels.
    Label 1 = Forget Set (Target)
    Label 0 = Retain Set
    """
    print(f"Loading dataset: {dataset_name} ({forget_subset} vs {retain_subset})")
    forget = load_dataset(dataset_name, forget_subset)
    retain = load_dataset(dataset_name, retain_subset)

    def concatenate_question_answer(example):
        example['qa'] = example['question'] + '\n' + example['answer']
        return example

    forget = forget.map(concatenate_question_answer)
    retain = retain.map(concatenate_question_answer)

    # Balance datasets if needed, or select a subset of retain
    if len(retain['train']) > len(forget['train']):
        retain = retain['train'].shuffle(seed=seed).select(range(len(forget['train'])))
    else:
        retain = retain['train']
        
    forget = forget['train']

    # Assign labels for the audit (Y)
    forget = forget.add_column('label', [1] * forget.shape[0])
    retain = retain.add_column('label', [0] * retain.shape[0])

    dataset = concatenate_datasets([forget, retain])
    
    # We use the FULL dataset for the audit (Train+Test split for RINE training)
    # RINE needs a train set to learn the decoders.
    dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    return dataset

# 2. Activation Extraction

def get_activations(model_path, dataset, save_path, batch_size=1, device='cuda'):
    """
    Loads a model, extracts last-layer representations, saves to disk, and unloads model.
    """
    if os.path.exists(save_path):
        print(f"Loading cached activations from {save_path}")
        data = np.load(save_path, allow_pickle=True).item()
        return data['activations'], data['labels']

    print(f"Extracting activations for model: {model_path}")
    
    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    model.eval()

    activations_list = []
    labels_list = []

    # Process both splits (train and test) to get full representations
    # We combine them because we want to audit the *representations* themselves.
    full_data = concatenate_datasets([dataset['train'], dataset['test']])

    with torch.no_grad():
        for i in tqdm(range(0, len(full_data), batch_size)):
            batch = full_data[i:i+batch_size]
            texts = batch['qa']
            labels = batch['label']

            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            
            # Get hidden states
            outputs = model(**inputs, output_hidden_states=True)
            
            # Extract last layer, last token
            # hidden_states is a tuple of (layer_0, ..., layer_N)
            # We take [-1] for last layer
            last_layer = outputs.hidden_states[-1] 
            
            # Pull last token features
            # For simplicity assuming left-padding or single batch:
            last_token_acts = last_layer[:, -1, :].cpu().numpy()
            
            activations_list.append(last_token_acts)
            labels_list.extend(labels)

    activations = np.concatenate(activations_list, axis=0)
    labels = np.array(labels_list)

    # Save to disk
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, {'activations': activations, 'labels': labels})
    print(f"Saved activations to {save_path}")
    
    # Cleanup to free GPU
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return activations, labels

# 3. Main Audit Loop

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Prepare Data
    dataset = load_and_preprocess_dataset(
        args.dataset_name, 
        args.forget_subset, 
        args.retain_subset
    )

    # 2. Get Representations (Z)
    # We use a naming convention for cache files to avoid re-running heavy inference
    base_cache = os.path.join(args.cache_dir, f"base_{args.forget_subset}.npy")
    unlearned_cache = os.path.join(args.cache_dir, f"unlearned_{args.exp_name}.npy")

    Z_base, Y = get_activations(args.base_model, dataset, base_cache, device=device)
    Z_unlearned, Y_u = get_activations(args.unlearned_model, dataset, unlearned_cache, device=device)

    assert np.array_equal(Y, Y_u), "Labels mismatch between base and unlearned extraction!"
    assert len(Z_base) == len(Z_unlearned), "Sample count mismatch!"

    # 3. Run RINE Audit
    print("\n" + "="*40)
    print("STARTING INFORMATION DECOMPOSITION AUDIT")
    print("="*40)
    
    input_dim = Z_base.shape[1]
    rine = RINE(dim_base=input_dim, dim_unlearned=input_dim, device=device)
    
    # Train estimator
    # Note: RINE uses the data to learn the dependency structure. 
    print(f"Training RINE on {len(Z_base)} samples...")
    metrics = rine.train_estimator(Z_base, Z_unlearned, Y, beta=args.beta, epochs=args.epochs)
    
    # 4. Report Results
    print("\n--- Audit Results (Information Theoretic) ---")
    print(f"Dataset: {args.dataset_name} ({args.forget_subset})")
    print("-" * 30)
    print(f"Unlearned Knowledge (Unique to Base): {metrics['Unlearned Knowledge']:.4f} bits")
    print(f"Residual Knowledge (Shared Risk):     {metrics['Residual Knowledge']:.4f} bits")
    print("-" * 30)
    
    # 5. Compute Risk Scores for Abstention (Section 6)
    print("\n--- Inference Risk Analysis ---")
    # Get probabilities from the trained probes
    logits_b = metrics['logits_base'].flatten()
    logits_u = metrics['logits_unlearned'].flatten()
    
    p1 = 1 / (1 + np.exp(-logits_b)) # Sigmoid
    p2 = 1 / (1 + np.exp(-logits_u))
    
    # Risk Score formula (Algorithm 1)
    # Risk = Average_Prob * Agreement
    risk_scores = 0.5 * (p1 + p2) * (1 - np.abs(p1 - p2))
    
    # Analyze risk on actual forgotten samples
    forget_indices = (Y == 1)
    avg_risk = risk_scores[forget_indices].mean()
    
    print(f"Avg Risk Score on Forget Set: {avg_risk:.4f} (Lower is safer)")
    print(f"Avg Risk Score on Retain Set: {risk_scores[~forget_indices].mean():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model Config
    parser.add_argument("--base_model", type=str, required=True, help="HuggingFace path to base model")
    parser.add_argument("--unlearned_model", type=str, required=True, help="HuggingFace path to unlearned model")
    
    # Data Config
    parser.add_argument("--dataset_name", type=str, default="locuslab/TOFU", help="Dataset path")
    parser.add_argument("--forget_subset", type=str, default="forget10", help="Subset to forget")
    parser.add_argument("--retain_subset", type=str, default="retain90", help="Subset to retain")
    
    # Audit Config
    parser.add_argument("--cache_dir", type=str, default="./activations", help="Where to save Z representations")
    parser.add_argument("--exp_name", type=str, default="default", help="Unique name for this run (for caching)")
    parser.add_argument("--beta", type=float, default=5.0, help="Lagrangian multiplier for RINE")
    parser.add_argument("--epochs", type=int, default=1000, help="Optimization steps for RINE")
    
    args = parser.parse_args()
    main(args)