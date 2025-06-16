import random
import re
import json
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from scipy.stats import norm
import pandas as pd
import os
from torch.cuda import OutOfMemoryError
from typing import Tuple, List, Dict, Any, Optional
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("math_eval.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "models": [ # ANY MODEL YOU WANT
        "openai-community/gpt2",
        "HuggingFaceTB/SmolLM-1.7B-Instruct",
        "google/gemma-2-2b-it"
    ],
    "num_questions": 5000,
    "max_tokens": 16,
    "batch_size": 8,
    "timeout": 10,
    "difficulty": "mixed",
    "output_dir": "results",
    "device": None
}

# Determine if we are in a Jupyter/Colab environment
IN_NOTEBOOK = 'ipykernel' in sys.modules

# Create a class to hold configuration - works in both script and notebook contexts
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Parse arguments or use defaults based on environment
if IN_NOTEBOOK:
    # In notebook, use default configuration
    args = Args(**DEFAULT_CONFIG)
    logger.info("Running in notebook environment, using default configuration")
else:
    # In script environment, use argparse
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate LLMs on basic arithmetic operations')
    parser.add_argument('--models', nargs='+', default=DEFAULT_CONFIG["models"],
                        help='List of models to evaluate')
    parser.add_argument('--num_questions', type=int, default=DEFAULT_CONFIG["num_questions"],
                        help='Number of questions to evaluate')
    parser.add_argument('--max_tokens', type=int, default=DEFAULT_CONFIG["max_tokens"],
                        help='Maximum tokens for model generation')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG["batch_size"],
                        help='Batch size for evaluation')
    parser.add_argument('--timeout', type=int, default=DEFAULT_CONFIG["timeout"],
                        help='Timeout in seconds for model inference')
    parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard', 'mixed'],
                        default=DEFAULT_CONFIG["difficulty"], help='Math problem difficulty')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_CONFIG["output_dir"],
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default=DEFAULT_CONFIG["device"],
                        help='Device to run on (default: auto-detect)')
    args = parser.parse_args()

# Set device
if args.device:
    DEVICE = args.device
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# Function to update configuration in notebook environment
def update_config(**kwargs):
    """Update configuration parameters (for use in notebooks)"""
    for key, value in kwargs.items():
        if hasattr(args, key):
            setattr(args, key, value)
            logger.info(f"Updated {key} to {value}")
        else:
            logger.warning(f"Unknown configuration parameter: {key}")

    # Re-set device if it was updated
    global DEVICE
    if "device" in kwargs:
        DEVICE = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {DEVICE}")

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Define difficulty levels for math problems
DIFFICULTY_RANGES = {
    'easy': {'num_range': (0, 10), 'operations': ['+', '-', '*', '/']},
    'medium': {'num_range': (10, 100), 'operations': ['+', '-', '*', '/']},
    'hard': {'num_range': (100, 1000), 'operations': ['+', '-', '*', '/', '**']},
    'mixed': None  # Will be handled specially
}

def get_difficulty_params(difficulty: str) -> dict:
    """Get parameters for the specified difficulty level."""
    if difficulty == 'mixed':
        # Randomly choose a difficulty
        difficulty = random.choice(['easy', 'medium', 'hard'])

    return DIFFICULTY_RANGES[difficulty]

def generate_problem(difficulty: str = 'easy') -> Tuple[str, str, str]:
    """Generate a math problem with the specified difficulty.

    Returns:
        Tuple containing (question, expected_answer, operation)
    """
    params = get_difficulty_params(difficulty)
    num_range = params['num_range']
    operations = params['operations']

    op = random.choice(operations)

    if op == '/':
        b = random.randint(1, num_range[1])
        # Make sure division results in an integer
        a = b * random.randint(1, num_range[1] // b)
        result = a // b
        question = f"{a} / {b}"  # Only the mathematical expression
        return question, str(result), op

    elif op == '*':
        a = random.randint(num_range[0], num_range[1])
        b = random.randint(num_range[0], num_range[1])
        result = a * b
        question = f"{a} * {b}"  # Only the mathematical expression
        return question, str(result), op

    elif op == '+':
        a = random.randint(num_range[0], num_range[1])
        b = random.randint(num_range[0], num_range[1])
        result = a + b
        question = f"{a} + {b}"  # Only the mathematical expression
        return question, str(result), op

    elif op == '-':
        a = random.randint(num_range[0], num_range[1])
        b = random.randint(num_range[0], min(a, num_range[1]))  # Ensure b <= a for simpler problems
        result = a - b
        question = f"{a} - {b}"  # Only the mathematical expression
        return question, str(result), op

    elif op == '**':
        a = random.randint(2, 10)
        b = random.randint(2, 3)  # Keep exponents small
        result = a ** b
        question = f"{a} ** {b}"  # Only the mathematical expression
        return question, str(result), op


def load_model_and_tokenizer(model_name: str) -> Tuple[Any, Any]:
    """Load model and tokenizer with error handling."""
    try:
        start_time = time.time()
        logger.info(f"Loading {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        model.eval()

        load_time = time.time() - start_time
        logger.info(f"Loaded {model_name} in {load_time:.2f} seconds")

        return tokenizer, model
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {str(e)}")
        raise

def extract_answer(response: str) -> str:
    """Extract the numeric answer from model response using regex patterns."""
    # Try to find a numeric answer in the response
    patterns = [
        r"answer is (\d+)",
        r"result is (\d+)",
        r"equals (\d+)",
        r"= (\d+)",
        r"(\d+)",  # Fallback to any number
    ]

    for pattern in patterns:
        matches = re.search(pattern, response.lower())
        if matches:
            return matches.group(1)

    return ""

def ask_model(tokenizer: Any, model: Any, question: str) -> str:
    """Query the model with error handling and timeout."""
    try:
        inputs = tokenizer(question, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            # Set a timeout for generation
            start_time = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                pad_token_id=tokenizer.eos_token_id
            )

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the answer part
        answer_text = full_response[len(question):].strip()

        # Try to extract the numeric answer
        answer = extract_answer(answer_text)

        return answer

    except OutOfMemoryError:
        logger.error(f"Out of memory error when processing: {question}")
        torch.cuda.empty_cache()
        return ""
    except Exception as e:
        logger.error(f"Error during model inference: {str(e)}")
        return ""

def calculate_confidence_interval(correct: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for binomial proportion."""
    if total == 0:
        return 0.0, 0.0

    p = correct / total
    z = norm.ppf(1 - (1 - confidence) / 2)
    margin = z * (p * (1 - p) / total) ** 0.5

    return max(0, p - margin), min(1, p + margin)

def evaluate_models() -> Dict[str, Dict[str, Any]]:
    """Evaluate all models and return results."""
    results = {}

    for model_name in args.models:
        logger.info(f"\nEvaluating {model_name}...")

        try:
            tokenizer, model = load_model_and_tokenizer(model_name)
        except Exception:
            logger.warning(f"Skipping {model_name} due to loading failure")
            results[model_name] = {
                "correct": 0,
                "incorrect": args.num_questions,
                "accuracy": 0.0,
                "ci_low": 0.0,
                "ci_high": 0.0,
                "by_operation": {op: {"correct": 0, "total": 0, "accuracy": 0.0}
                               for op in ['+', '-', '*', '/', '**']}
            }
            continue

        correct = 0
        incorrect = 0
        by_operation = {op: {"correct": 0, "total": 0} for op in ['+', '-', '*', '/', '**']}

        for _ in tqdm(range(args.num_questions)):
            question, expected_answer, operation = generate_problem(args.difficulty)

            try:
                answer = ask_model(tokenizer, model, question)

                # Update operation-specific stats
                by_operation[operation]["total"] += 1

                if answer.isdigit() and answer == expected_answer:
                    correct += 1
                    by_operation[operation]["correct"] += 1
                else:
                    incorrect += 1

            except Exception as e:
                logger.error(f"Error evaluating {question}: {str(e)}")
                incorrect += 1

        # Calculate accuracies and confidence intervals
        accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0
        ci_low, ci_high = calculate_confidence_interval(correct, correct + incorrect)

        # Calculate operation-specific accuracies
        for op in by_operation:
            op_total = by_operation[op]["total"]
            op_correct = by_operation[op]["correct"]
            by_operation[op]["accuracy"] = op_correct / op_total if op_total > 0 else 0

        results[model_name] = {
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": accuracy,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "by_operation": by_operation
        }

        # Save intermediate results
        save_results(results)

    return results

def save_results(results: Dict[str, Dict[str, Any]]) -> None:
    """Save results to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(args.output_dir, f"math_eval_results_{timestamp}.json")

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {filename}")

def plot_results(results: Dict[str, Dict[str, Any]]) -> None:
    """Create visualizations of the results."""
    # Set the style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})

    # 1. Overall accuracy bar chart with confidence intervals
    plt.figure(figsize=(14, 8))

    model_names = list(results.keys())
    accuracies = [results[m]["accuracy"] * 100 for m in model_names]
    ci_lows = [results[m]["ci_low"] * 100 for m in model_names]
    ci_highs = [results[m]["ci_high"] * 100 for m in model_names]

    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)[::-1]
    model_names = [model_names[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    ci_lows = [ci_lows[i] for i in sorted_indices]
    ci_highs = [ci_highs[i] for i in sorted_indices]

    # Calculate error bar values
    yerr_low = [acc - low for acc, low in zip(accuracies, ci_lows)]
    yerr_high = [high - acc for acc, high in zip(accuracies, ci_highs)]

    # Clean up model names for display
    display_names = [name.split('/')[-1] for name in model_names]

    plt.bar(display_names, accuracies, color='skyblue', edgecolor='black')
    plt.errorbar(display_names, accuracies, yerr=[yerr_low, yerr_high], fmt='o', color='black', capsize=5)

    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Model Accuracy on Math Problems ({args.num_questions} questions each, {args.difficulty} difficulty)')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)

    for i, v in enumerate(accuracies):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'overall_accuracy.png'), dpi=300)

    # 2. Stacked bar chart showing correct vs incorrect
    plt.figure(figsize=(14, 8))

    model_names = list(results.keys())
    # Sort by accuracy
    sorted_indices = np.argsort([results[m]["accuracy"] for m in model_names])[::-1]
    model_names = [model_names[i] for i in sorted_indices]
    display_names = [name.split('/')[-1] for name in model_names]

    correct_vals = [results[m]["correct"] for m in model_names]
    incorrect_vals = [results[m]["incorrect"] for m in model_names]

    plt.bar(display_names, correct_vals, color='green', alpha=0.7, label='Correct')
    plt.bar(display_names, incorrect_vals, bottom=correct_vals, color='red', alpha=0.7, label='Incorrect')

    plt.xlabel('Model')
    plt.ylabel('Number of Questions')
    plt.title(f'Model Performance on Math Problems ({args.num_questions} questions each)')
    plt.xticks(rotation=45, ha='right')
    plt.legend()

    # Add accuracy percentages on top of bars
    total_height = [c + i for c, i in zip(correct_vals, incorrect_vals)]
    for i, (c, t) in enumerate(zip(correct_vals, total_height)):
        plt.text(i, t/2, f"{c/t*100:.1f}%", ha='center', va='center', color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'correct_vs_incorrect.png'), dpi=300)

    # 3. Heatmap of operation-specific accuracy
    plt.figure(figsize=(14, 10))

    # Prepare data for heatmap
    operations = ['+', '-', '*', '/']
    if args.difficulty in ['hard', 'mixed']:
        operations.append('**')

    heatmap_data = []
    for model in model_names:
        model_data = []
        for op in operations:
            if op in results[model]["by_operation"]:
                acc = results[model]["by_operation"][op]["accuracy"] * 100
                model_data.append(acc)
            else:
                model_data.append(0)
        heatmap_data.append(model_data)

    # Create DataFrame for heatmap
    df = pd.DataFrame(heatmap_data, columns=operations, index=display_names)

    # Plot heatmap
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".1f", cbar_kws={'label': 'Accuracy (%)'})
    plt.title('Accuracy by Operation Type (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'operation_heatmap.png'), dpi=300)

    # Show all plots
    plt.show()

def main() -> None:
    logger.info(f"Starting evaluation with {len(args.models)} models and {args.num_questions} questions each")
    logger.info(f"Difficulty level: {args.difficulty}")

    start_time = time.time()
    results = evaluate_models()
    total_time = time.time() - start_time

    logger.info(f"Evaluation completed in {total_time:.2f} seconds")

    # Save final results
    save_results(results)

    # Plot results
    plot_results(results)

    return results

# For notebook usage, provide a function to run with custom parameters
def run_evaluation(**kwargs):
    """Run evaluation with custom parameters (for use in notebooks)"""
    # Update configuration if parameters provided
    if kwargs:
        update_config(**kwargs)

    # Run evaluation
    return main()

if __name__ == "__main__":
    main()
