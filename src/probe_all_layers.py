import torch
import transformer_lens
from transformer_lens import HookedTransformer
from datasets import load_dataset, concatenate_datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
import scipy.stats
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # for creating DataFrame for heatmap

def load_and_preprocess_dataset(dataset_name, forget_subset, retain_subset, seed=42, test_size=0.1):
    """Loads and preprocesses the dataset (same as before)."""
    forget = load_dataset(dataset_name, forget_subset)
    retain = load_dataset(dataset_name, retain_subset)

    def concatenate_question_answer(example):
        example['qa'] = example['question'] + '\n' + example['answer']
        return example

    forget = forget.map(concatenate_question_answer)
    retain = retain.map(concatenate_question_answer)

    retain = retain['train'].shuffle(seed=seed).select(range(len(forget['train'])))
    forget = forget['train']

    forget = forget.add_column('label', [0] * forget.shape[0])
    retain = retain.add_column('label', [1] * retain.shape[0])

    dataset = concatenate_datasets([forget, retain])
    dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    return dataset

def load_transformer_model(model_name, device, hf_model_name=None):
    """Loads a HookedTransformer model (same as before)."""
    if hf_model_name:
        hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name, device_map='auto')
    else:
        hf_model = None

    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        hf_model=hf_model
    )
    return model

def get_activations(model, data, layer_num, layer_type='mlp', device='cpu', save_path=None):
    """Extracts activations (modified to include 'resid_post')."""
    model.to(device)
    activations_list = []

    for sample in data:
        text = sample['qa']
        tokens = model.tokenizer(text, return_tensors='pt').to(device)['input_ids']
        with torch.no_grad():
            output, cache = model.run_with_cache(tokens)
            if layer_type == 'mlp':
                hook_name = f"blocks.{layer_num}.mlp.hook_post"
            elif layer_type == 'attn':
                hook_name = f"blocks.{layer_num}.attn.hook_pattern"
            elif layer_type == 'residual':
                hook_name = f'blocks.{layer_num}.hook_resid_post'
            elif layer_type == 'resid_post': # Added 'resid_post' layer type
                hook_name = f'blocks.{layer_num}.hook_resid_post'
            else:
                raise ValueError(f"Invalid layer_type: {layer_type}. Must be 'mlp', 'attn', 'residual', or 'resid_post'.")

            layer_activations = cache[hook_name]
            last_token_activations = layer_activations[:, -1, :].squeeze().cpu().numpy()
            activations_list.append(last_token_activations)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, np.array(activations_list))
        print(f"Activations saved to {save_path}")

    return activations_list

def load_activations(load_path):
    """Loads activations from a saved numpy file (same as before)."""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Activations file not found at {load_path}")
    activations_array = np.load(load_path)
    return activations_array.tolist()

def train_linear_probe(train_activations, train_labels, test_activations, test_labels):
    """Trains a linear probe (same as before)."""
    probe = LogisticRegression()
    probe.fit(train_activations, train_labels)
    test_predictions = probe.predict(test_activations)
    test_probabs = probe.predict_proba(test_activations)[:, 1]
    score = roc_auc_score(test_labels, test_probabs)
    accuracy = accuracy_score(test_labels, test_predictions)
    corr = scipy.stats.spearmanr(test_labels, test_predictions)
    return score, accuracy, corr




def plot_heatmap(results, experiment_name):
    """Plots a heatmap of Spearman correlations."""
    layer_types = list(results.keys())
    layer_nums = list(results[layer_types[0]].keys())
    data = []
    for layer_type in layer_types:
        correlations = [results[layer_type][layer] for layer in layer_nums]
        data.append(correlations)

    df = pd.DataFrame(data, index=layer_types, columns=layer_nums)

    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="viridis") # annot=True to show values, fmt to format numbers
    plt.xlabel("Layer Number")
    plt.ylabel("Layer Type")
    plt.title(f"Spearman  Heatmap - {experiment_name}")
    plt.savefig(f"heatmap_{experiment_name}.png") # Save the heatmap
    plt.show()

def plot_lineplot(results, experiment_name, metric=None):
    """Plots a line plot of Spearman correlations for different layer types."""
    layer_types = list(results.keys())
    layer_nums = list(results[layer_types[0]].keys())

    plt.figure(figsize=(12, 6)) # Adjust figure size for better readability

    for layer_type in layer_types:
        correlations = [results[layer_type][layer] for layer in layer_nums]
        plt.plot(layer_nums, correlations, label=layer_type) # Plot each layer type as a line

    plt.xlabel("Layer Number")
    plt.ylabel(metric)
    plt.title(f"{metric} vs. Layer Number - {experiment_name}")
    plt.legend() # Show legend to identify layer types
    plt.grid(True, linestyle='--', alpha=0.6) # Add grid for better readability
    plt.savefig(f"lineplot_{experiment_name}_{metric}.png") # Save the line plot
    plt.show()


def main(config):
    """Main function to run linear probes across layers and layer types."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    dataset = load_and_preprocess_dataset(
        dataset_name=config['dataset_name'],
        forget_subset=config['forget_subset'],
        retain_subset=config['retain_subset']
    )

    model = load_transformer_model(
        model_name=config['model_name'],
        device=device,
        hf_model_name=config.get('hf_model_name')
    )

    num_layers = model.cfg.n_layers # Get number of layers from model config
    layer_types = ['mlp', 
                #    'attn', 
                   'residual'] # List of layer types to test
    results = {} # Dictionary to store results: {layer_type: {layer_num: spearman_corr}}
    correlations = {}
    accuracies = {}
    auc_scores = {}

    activations_dir = config.get('activations_dir', 'activations')
    os.makedirs(activations_dir, exist_ok=True)

    train_labels = [i['label'] for i in dataset['train']]
    test_labels = [i['label'] for i in dataset['test']]

    for layer_type in layer_types:
        results[layer_type] = {}
        correlations[layer_type] = {}
        accuracies[layer_type] = {}
        auc_scores[layer_type] = {}
        for layer_num in range(num_layers):
            print(f"\nProcessing Layer: {layer_num}, Layer Type: {layer_type}")

            experiment_name = f"{config['experiment_name']}_layer{layer_num}_{layer_type}"
            train_activations_path = os.path.join(activations_dir, f"{experiment_name}_train_activations.npy")
            test_activations_path = os.path.join(activations_dir, f"{experiment_name}_test_activations.npy")

            if config.get('load_activations_from_disk', False) and os.path.exists(train_activations_path) and os.path.exists(test_activations_path):
                print("Loading activations from disk...")
                train_activations = load_activations(train_activations_path)
                test_activations = load_activations(test_activations_path)
            else:
                print("Extracting activations...")
                train_activations = get_activations(
                    model, dataset['train'], layer_num, layer_type, device, save_path=train_activations_path
                )
                test_activations = get_activations(
                    model, dataset['test'], layer_num, layer_type, device, save_path=test_activations_path
                )

            print("Training linear probe...")
            roc_auc, accuracy, corr = train_linear_probe(
                train_activations,
                train_labels,
                test_activations,
                test_labels
            )

            correlations[layer_type][layer_num] = corr.correlation # Store Spearman correlation
            accuracies[layer_type][layer_num] = accuracy
            auc_scores[layer_type][layer_num] = roc_auc
            # print(f"Layer {layer_num}, {layer_type} - Spearman Correlation: {corr.correlation:.4f}")

    # Create Heatmap
    # plot_heatmap(results, config['experiment_name'])
    plot_lineplot(correlations, config['experiment_name'], metric='correlation')
    plot_lineplot(accuracies, config['experiment_name'], metric='accuracy')
    plot_lineplot(auc_scores, config['experiment_name'], metric='auc score')



if __name__ == "__main__":
    # Example Configuration
    config = {
        'experiment_name': 'tofu_llama3.2-1b-rmu_forget10', # Experiment name for file saving and plot title
        'dataset_name': "locuslab/TOFU",
        'forget_subset': "forget10",
        'retain_subset': "retain90",
        'model_name': "meta-llama/Llama-3.2-1B-Instruct",
        'hf_model_name': '/auditing_unlearning/open-unlearning/experiments/saves/unlearn/llama-3.2-1b-rmu',
        #'layer_num': 15, # No longer needed as we iterate through all layers
        #'layer_type': 'mlp', # No longer needed as we iterate through all layer_types
        'activations_dir': 'activations_all_layers', # Separate directory for all layers activations
        'load_activations_from_disk': True
    }

    main(config)