#!/usr/bin/env python3
"""
Sequential Batch Testing for vLLM Inference

Reads prompts from JSON config and runs them sequentially against specified models.
"""

import json
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import subprocess

def load_config(config_file):
    """Load configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{config_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}")
        sys.exit(1)

def filter_prompts(prompts, category=None, priority=None):
    """Filter prompts by category and/or priority."""
    # Since prompts only have 'id' and 'prompt' fields, return all prompts
    # Category and priority filtering is no longer supported
    if category or priority:
        print("Warning: Category and priority filtering not supported with simplified prompt format")
    return prompts

def run_single_prompt(prompt_data, model, settings, output_dir):
    """Run a single prompt against a single model."""
    print(f"\n{'='*60}")
    print(f"Running: {prompt_data['id']} | Model: {model}")
    print(f"Prompt: {prompt_data['prompt'][:100]}...")
    print(f"{'='*60}")
    
    # Build command
    cmd = [
        'python', 'vllm_inference.py',
        prompt_data['prompt'],
        '--model', model,
        '--prompter-api-key', settings['api_key'],
        '--index', settings['index_dir']
    ]
    
    # Add optional parameters
    params = settings.get('default_params', {})
    if params.get('expand'):
        cmd.append('--expand')
    if params.get('cited_spans'):
        cmd.append('--cited-spans')
    if params.get('preset'):
        cmd.extend(['--preset', params['preset']])
    if params.get('temperature'):
        cmd.extend(['--temperature', str(params['temperature'])])
    if params.get('top_p'):
        cmd.extend(['--top-p', str(params['top_p'])])
    if params.get('max_tokens'):
        cmd.extend(['--max-new-tokens', str(params['max_tokens'])])
    
    # Add output JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/{prompt_data['id']}_{model}_{timestamp}.json"
    cmd.extend(['--output-json', output_file])
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ Success: Output saved to {output_file}")
            return True, output_file
        else:
            print(f"❌ Error: {result.stderr}")
            return False, None
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout: Command took longer than 5 minutes")
        return False, None
    except Exception as e:
        print(f"💥 Exception: {e}")
        return False, None

def main():
    parser = argparse.ArgumentParser(description='Sequential batch testing for vLLM inference')
    parser.add_argument('config_file', help='JSON configuration file')
    parser.add_argument('--category', help='Filter by category (not supported with simplified format)')
    parser.add_argument('--priority', help='Filter by priority (not supported with simplified format)')
    parser.add_argument('--models', nargs='+', help='Override models from config')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be run without executing')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_file)
    settings = config['settings']
    prompts = config['prompts']
    
    # Filter prompts
    filtered_prompts = filter_prompts(prompts, args.category, args.priority)
    
    if not filtered_prompts:
        print("No prompts match the specified filters.")
        return
    
    # Use models from args or config
    models = args.models if args.models else settings['models']
    
    # Create output directory
    output_dir = settings.get('output_dir', 'results')
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"📋 Found {len(filtered_prompts)} prompts to test")
    print(f"🤖 Testing with {len(models)} models: {', '.join(models)}")
    print(f"📁 Output directory: {output_dir}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - Commands that would be executed:")
        for prompt_data in filtered_prompts:
            for model in models:
                print(f"  - {prompt_data['id']} × {model}: {prompt_data['prompt'][:50]}...")
        return
    
    # Run tests
    total_tests = len(filtered_prompts) * len(models)
    current_test = 0
    successful_tests = 0
    
    results_summary = []
    
    for prompt_data in filtered_prompts:
        for model in models:
            current_test += 1
            print(f"\n📊 Progress: {current_test}/{total_tests}")
            
            success, output_file = run_single_prompt(prompt_data, model, settings, output_dir)
            
            results_summary.append({
                'prompt_id': prompt_data['id'],
                'model': model,
                'success': success,
                'output_file': output_file,
                'timestamp': datetime.now().isoformat()
            })
            
            if success:
                successful_tests += 1
    
    # Save summary
    summary_file = f"{output_dir}/batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'config_file': args.config_file,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'filters': {
                'category': args.category,
                'priority': args.priority
            },
            'results': results_summary
        }, f, indent=2)
    
    print(f"\n🎉 Batch testing complete!")
    print(f"✅ Successful: {successful_tests}/{total_tests}")
    print(f"📄 Summary saved to: {summary_file}")

if __name__ == '__main__':
    main()
