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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


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


def evaluate_multiple_choice(model, processor, audio_file, candidate_captions, 
                           audio_folder, device):
    """Evaluate multiple choice: given an audio and caption options, let model select the best one."""
    
    # Load audio
    audio_path = os.path.join(audio_folder, audio_file)
    audio = load_audio_file(audio_path, processor)
    
    if audio is None:
        return -1, 0.0, ""  # No prediction, 0 confidence, empty response
    
    # Create multiple choice prompt
    prompt_template = "<|audio_bos|><|AUDIO|><|audio_eos|>Given the audio above, which of the following captions best describes it?\n"
    
    # Add caption options
    for i, caption in enumerate(candidate_captions):
        prompt_template += f"{i+1}. {caption}\n"
    
    prompt_template += "Answer with the number (1, 2, or 3) of the best matching caption:"
    
    try:
        inputs = processor(text=prompt_template, audios=audio, return_tensors="pt")
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_length=512)
            gen_ids = output_ids[:, inputs.input_ids.size(1):]
            response = processor.decode(gen_ids, skip_special_tokens=True)[0]
        
        response = response.strip()
        
        # Extract the predicted choice (1, 2, or 3)
        prediction = -1
        confidence = 0.0
        
        # Try to extract number from response
        import re
        numbers = re.findall(r'\d+', response)
        if numbers:
            predicted_num = int(numbers[0])
            if 1 <= predicted_num <= len(candidate_captions):
                prediction = predicted_num - 1  # Convert to 0-based index
                confidence = 1.0  # High confidence if we got a valid number
        
        return prediction, confidence, response
        
    except Exception as e:
        print(f"Error in multiple choice evaluation: {e}")
        return -1, 0.0, ""


def evaluate_audio_caption_matching(model, processor, benchmark_data, audio_folder, device, output_file):
    """Evaluate the model's ability to match audios with their captions using multiple choice."""
    
    all_predictions = []
    all_ground_truth = []
    all_confidences = []
    total_questions = 0
    correct_questions = 0
    
    # Initialize the output file with metadata
    output_data = {
        "metadata": {
            "model_name": "Qwen2AudioForConditionalGeneration",
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

        print(f"Selected audio file: {selected_audio_file}")
        print(f"Selected caption: {selected_caption}")
        print(f"Selected correct answer: {selected_correct_answer}")
        
        # Evaluate the selected audio file
        prediction, confidence, response = evaluate_multiple_choice(
            model, processor, selected_audio_file, candidate_captions, audio_folder, device
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
            ]
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
    parser = argparse.ArgumentParser(description='Audio-Caption Multiple Choice Evaluation (Direct Selection)')
    parser.add_argument('--benchmark-csv', type=str, 
                       default='audio_eval/CompA_order/CompA_order_benchmark.csv',
                       help='Path to the benchmark CSV file')
    parser.add_argument('--audio-folder', type=str, 
                       default='audio_eval/CompA_order/CompA_order_files',
                       help='Path to the folder containing audio files')
    parser.add_argument('--output-file', type=str, default='audio_caption_direct_results.json',
                       help='Output file to save results')
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2-Audio-7B',
                       help='HuggingFace model name')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda, cuda:0, cuda:1, cpu)')
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
    
    # Load benchmark data
    print("Loading benchmark data...")
    benchmark_data = pd.read_csv(args.benchmark_csv)
    
    if args.max_samples:
        benchmark_data = benchmark_data.head(args.max_samples)
    
    print(f"Evaluating {len(benchmark_data)} samples...")
    
    # Run evaluation
    results = evaluate_audio_caption_matching(
        model=model,
        processor=processor,
        benchmark_data=benchmark_data,
        audio_folder=args.audio_folder,
        device=device,
        output_file=args.output_file
    )
    
    # Print results
    print(f"\n=== Direct Multiple Choice Evaluation Results ===")
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