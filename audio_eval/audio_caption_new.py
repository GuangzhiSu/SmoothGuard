import argparse
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import librosa
import shortuuid
from io import BytesIO
from urllib.request import urlopen
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_roberta_embeddings, make_clustering
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.cluster import KMeans
import re


def apply_audio_random_smoothing(audio, sigma=0.1, num_copy=5):
    """Apply random smoothing to audio by adding Gaussian noise."""
    if num_copy == 0:
        return [audio]
    
    noisy_audios = []
    for _ in range(num_copy + 1):  # +1 for the original
        if _ == 0:
            noisy_audios.append(audio)
        else:
            noise = np.random.normal(0, sigma, audio.shape)
            noisy_audio = audio + noise
            noisy_audios.append(noisy_audio)
    
    return noisy_audios


def load_audio_file(audio_path, processor):
    """Load and preprocess audio file."""
    try:
        if audio_path.startswith('http'):
            # Load from URL
            audio, sr = librosa.load(BytesIO(urlopen(audio_path).read()), 
                                   sr=processor.feature_extractor.sampling_rate)
        else:
            # Load from local file
            audio, sr = librosa.load(audio_path, 
                                   sr=processor.feature_extractor.sampling_rate)
        return audio
    except Exception as e:
        print(f"Error loading audio {audio_path}: {e}")
        return None


def extract_prediction_from_response(response):
    """Extract prediction number from model response."""
    try:
        numbers = re.findall(r'\d+', response)
        if numbers:
            predicted_num = int(numbers[0])
            if 1 <= predicted_num <= 3:  # Valid range for our multiple choice
                return predicted_num - 1  # Convert to 0-based index
        return -1
    except:
        return -1


def cluster_predictions_with_embeddings(responses, roberta_tokenizer, roberta_model, n_clusters=2):
    """Cluster responses into n_clusters groups using RoBERTa embeddings and K-means."""
    if len(responses) < n_clusters:
        return extract_prediction_from_response(responses[0]) if responses else -1
    
    # Get embeddings for all responses
    embeddings = []
    valid_responses = []
    
    for response in responses:
        if response.strip():
            try:
                emb = get_roberta_embeddings(response, roberta_tokenizer, roberta_model)
                embeddings.append(emb)
                valid_responses.append(response)
            except Exception as e:
                print(f"Error getting embedding for response: {e}")
                continue
    
    if len(embeddings) < n_clusters:
        return extract_prediction_from_response(valid_responses[0]) if valid_responses else -1
    
    # Stack embeddings
    embeddings_array = np.vstack(embeddings)
    
    # Apply K-means clustering
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)
        
        # Find the largest cluster
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        largest_cluster_label = unique_labels[np.argmax(counts)]
        
        # Get responses from the largest cluster
        largest_cluster_responses = [valid_responses[i] for i, label in enumerate(cluster_labels) if label == largest_cluster_label]
        
        # Extract prediction from the first response in the largest cluster
        return extract_prediction_from_response(largest_cluster_responses[0]) if largest_cluster_responses else -1
        
    except Exception as e:
        print(f"Error in clustering: {e}")
        return extract_prediction_from_response(valid_responses[0]) if valid_responses else -1


def evaluate_multiple_choice_with_clustering(model, processor, audio_file, candidate_captions, 
                                           audio_folder, roberta_tokenizer, roberta_model, device,
                                           sigma=0.1, num_copy=5):
    """Evaluate multiple choice with random noise and clustering: given an audio, select the best caption from candidates."""
    
    # Load audio
    audio_path = os.path.join(audio_folder, audio_file)
    audio = load_audio_file(audio_path, processor)
    
    if audio is None:
        return -1, 0.0, ""  # No prediction, 0 confidence, empty response
    
    # Apply random smoothing to audio and generate multiple responses
    noisy_audios = apply_audio_random_smoothing(audio, sigma, num_copy)
    responses = []
    
    # Create multiple choice prompt
    prompt_template = "<|audio_bos|><|AUDIO|><|audio_eos|>Given the audio above, which of the following captions best describes it?\n"
    
    # Add caption options
    for i, caption in enumerate(candidate_captions):
        prompt_template += f"{i+1}. {caption}\n"
    
    prompt_template += "Answer with the number (1, 2, or 3) of the best matching caption:"
    
    for noisy_audio in noisy_audios:
        try:
            inputs = processor(text=prompt_template, audios=noisy_audio, return_tensors="pt")
            
            with torch.no_grad():
                # Remove cache_position if it exists in inputs to avoid compatibility issues
                generation_kwargs = {
                    'max_new_tokens': 10,
                    'do_sample': False
                }
                
                # Filter out unsupported arguments
                supported_inputs = {k: v for k, v in inputs.items() if k != 'cache_position'}
                
                output_ids = model.generate(**supported_inputs, **generation_kwargs)
                gen_ids = output_ids[0][inputs['input_ids'].shape[-1]:]
                response = processor.decode(gen_ids, skip_special_tokens=True)
            
            response = response.strip()
            if response:  # Only add non-empty responses
                responses.append(response)
                
        except Exception as e:
            print(f"Error generating response: {e}")
            continue
    
    if not responses:
        return -1, 0.0, ""
    
    # Cluster the responses into 2 groups and select from the larger cluster
    if len(responses) > 1:
        final_prediction = cluster_predictions_with_embeddings(responses, roberta_tokenizer, roberta_model, n_clusters=2)
    else:
        final_prediction = extract_prediction_from_response(responses[0])
    
    # Set confidence based on whether we got a valid prediction
    confidence = 1.0 if final_prediction != -1 else 0.0
    
    return final_prediction, confidence, responses[0] if responses else ""


def evaluate_audio_caption_matching(model, processor, benchmark_data, audio_folder, 
                                  roberta_tokenizer, roberta_model, device, 
                                  sigma=0.1, num_copy=5, output_file="results.json"):
    """Evaluate the model's ability to match audios with their captions using multiple choice with clustering."""
    
    all_predictions = []
    all_ground_truth = []
    all_confidences = []
    total_questions = 0
    correct_questions = 0
    
    # Initialize the output file with metadata
    output_data = {
        "metadata": {
            "evaluation_type": "clustering_based_multiple_choice",
            "model_name": "Qwen2AudioForConditionalGeneration",
            "clustering_method": "kmeans_clustering_with_roberta_embeddings",
            "sigma": sigma,
            "num_copy": num_copy,
            "total_questions": 0,
            "correct_questions": 0,
            "accuracy": 0.0
        },
        "questions": []
    }
    
    # Write initial structure to file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    for idx, row in tqdm(benchmark_data.iterrows(), total=len(benchmark_data), desc="Evaluating"):
        # Get all available captions for this row
        candidate_captions = []
        audio_files = []
        correct_answers = []  # Index of correct answer for each audio
        
        # Collect all available captions and their corresponding audio files
        # Only include valid (non-null, non-dash) entries
        if pd.notna(row['pair_caption']) and pd.notna(row['pair_file']) and row['pair_caption'] != '-' and row['pair_file'] != '-':
            candidate_captions.append(row['pair_caption'])
            audio_files.append(row['pair_file'])
            correct_answers.append(0)  # This caption corresponds to the first audio
        
        if pd.notna(row['reversed_pair_caption']) and pd.notna(row['reversed_pair_file']) and row['reversed_pair_caption'] != '-' and row['reversed_pair_file'] != '-':
            candidate_captions.append(row['reversed_pair_caption'])
            audio_files.append(row['reversed_pair_file'])
            correct_answers.append(1)  # This caption corresponds to the second audio
        
        if pd.notna(row['triplet_caption']) and pd.notna(row['triplet_file']) and row['triplet_caption'] != '-' and row['triplet_file'] != '-':
            candidate_captions.append(row['triplet_caption'])
            audio_files.append(row['triplet_file'])
            correct_answers.append(2)  # This caption corresponds to the third audio
        
        # Need at least 2 valid options for multiple choice
        if len(candidate_captions) < 2:
            print(f"Skipping row {idx}: only {len(candidate_captions)} valid options available")
            continue
        
        # Randomly select one audio-caption pair from this row
        import random
        random.seed(42 + idx)  # Use deterministic seed for reproducibility
        selected_index = random.randint(0, len(audio_files) - 1)
        
        # Get the selected audio and its corresponding caption
        selected_audio_file = audio_files[selected_index]
        selected_correct_answer = correct_answers[selected_index]
        selected_caption = candidate_captions[selected_index]
        
        # Evaluate the selected audio file
        prediction, confidence, response = evaluate_multiple_choice_with_clustering(
            model, processor, selected_audio_file, candidate_captions, audio_folder,
            roberta_tokenizer, roberta_model, device, sigma, num_copy
        )
        
        # Check if prediction is correct
        is_correct = 1 if prediction == selected_correct_answer else 0
        total_questions += 1
        if is_correct:
            correct_questions += 1
        
        all_predictions.append(prediction)
        all_ground_truth.append(selected_correct_answer)
        all_confidences.append(confidence)
        
        # Create question result
        question_result = {
            "row_index": idx,
            "question_id": f"row_{idx}_random_selection",
            "selected_audio_file": selected_audio_file,
            "selected_audio_index": selected_index,
            "candidate_captions": candidate_captions,
            "correct_answer_index": selected_correct_answer,
            "correct_answer_caption": selected_caption,
            "model_prediction_index": prediction,
            "model_prediction_caption": candidate_captions[prediction] if prediction >= 0 and prediction < len(candidate_captions) else "No valid prediction",
            "model_response": response,
            "confidence": confidence,
            "is_correct": bool(is_correct),
            "all_caption_options": [
                {"index": j, "caption": caption} for j, caption in enumerate(candidate_captions)
            ],
            "all_available_audios": [
                {"index": j, "file": audio_file, "caption": caption} 
                for j, (audio_file, caption) in enumerate(zip(audio_files, candidate_captions))
            ],
            "clustering_info": {
                "sigma": sigma,
                "num_copy": num_copy,
                "method": "kmeans_clustering_with_roberta_embeddings"
            }
        }
        
        # Read current file, append new question, and write back
        try:
            with open(output_file, 'r') as f:
                current_data = json.load(f)
            
            current_data["questions"].append(question_result)
            current_data["metadata"]["total_questions"] = total_questions
            current_data["metadata"]["correct_questions"] = correct_questions
            current_data["metadata"]["accuracy"] = correct_questions / total_questions if total_questions > 0 else 0.0
            
            with open(output_file, 'w') as f:
                json.dump(current_data, f, indent=2)
                
        except Exception as e:
            print(f"Error writing to file: {e}")
    
    # Compute final overall metrics
    accuracy = accuracy_score(all_ground_truth, all_predictions) if all_predictions else 0
    avg_confidence = np.mean([c for c in all_confidences if c > 0]) if all_confidences else 0
    
    # For precision, recall, F1 - treat as classification problem
    if all_predictions:
        precision, recall, f1, _ = precision_recall_fscore_support(all_ground_truth, all_predictions, 
                                                                  average='weighted', zero_division=0)
    else:
        precision = recall = f1 = 0
    
    # Update final metrics in the file
    try:
        with open(output_file, 'r') as f:
            final_data = json.load(f)
        
        final_data["metadata"].update({
            "overall_accuracy": accuracy,
            "average_confidence": avg_confidence,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "total_evaluations": len(all_predictions),
            "correct_predictions": sum(1 for p, gt in zip(all_predictions, all_ground_truth) if p == gt)
        })
        
        with open(output_file, 'w') as f:
            json.dump(final_data, f, indent=2)
            
    except Exception as e:
        print(f"Error updating final metrics: {e}")
    
    return {
        "overall_accuracy": accuracy,
        "average_confidence": avg_confidence,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "total_evaluations": len(all_predictions),
        "correct_predictions": sum(1 for p, gt in zip(all_predictions, all_ground_truth) if p == gt)
    }


def main():
    parser = argparse.ArgumentParser(description='Audio-Caption Multiple Choice Evaluation with Random Noise and Clustering')
    parser.add_argument('--benchmark-csv', type=str, 
                       default='audio_eval/CompA_order/CompA_order_benchmark.csv',
                       help='Path to the benchmark CSV file')
    parser.add_argument('--audio-folder', type=str, 
                       default='audio_eval/CompA_order/CompA_order_files',
                       help='Path to the folder containing audio files')
    parser.add_argument('--output-file', type=str, default='audio_caption_clustering_results.json',
                       help='Output file to save results')
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2-Audio-7B',
                       help='HuggingFace model name')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda, cuda:0, cuda:1, cpu)')
    parser.add_argument('--sigma', type=float, default=0.1,
                       help='Standard deviation for Gaussian noise')
    parser.add_argument('--num-copy', type=int, default=5,
                       help='Number of noisy copies for random smoothing')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to evaluate')
    
    args = parser.parse_args()
    
    # Handle device specification properly
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device.startswith('cuda:') and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    print(f"Using device: {device}")
    
    # Load model and processor
    print("Loading model and processor...")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model_name, 
                                                              trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model.eval()
    
    # Load RoBERTa for clustering
    print("Loading RoBERTa for response clustering...")
    from transformers import RobertaTokenizer, RobertaModel
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta_model = RobertaModel.from_pretrained('roberta-base').to(device)
    roberta_model.eval()
    
    # Load benchmark data
    print("Loading benchmark data...")
    benchmark_data = pd.read_csv(args.benchmark_csv)
    
    if args.max_samples:
        benchmark_data = benchmark_data.head(args.max_samples)
    
    print(f"Evaluating {len(benchmark_data)} samples...")
    print(f"Using random smoothing with sigma={args.sigma}, num_copy={args.num_copy}")
    print("Clustering responses into 2 groups and selecting from the larger cluster")
    
    # Run evaluation
    results = evaluate_audio_caption_matching(
        model=model,
        processor=processor,
        benchmark_data=benchmark_data,
        audio_folder=args.audio_folder,
        roberta_tokenizer=roberta_tokenizer,
        roberta_model=roberta_model,
        device=device,
        sigma=args.sigma,
        num_copy=args.num_copy,
        output_file=args.output_file
    )
    
    # Print results
    print(f"\n=== Clustering-based Multiple Choice Evaluation Results ===")
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    print(f"Average Confidence: {results['average_confidence']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Correct Predictions: {results['correct_predictions']}/{results['total_evaluations']}")
    
    # Save detailed results
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("Evaluation completed!")


if __name__ == "__main__":
    main() 