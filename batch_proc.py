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


def run_single_prompt(prompt_data, model, settings, output_dir):
    """Run a single prompt against a single model and return the response."""
    print(f"\n{'='*60}")
    print(f"Running: {prompt_data['id']} | Model: {model}")
    print(f"Prompt: {prompt_data['prompt'][:100]}...")
    print(f"{'='*60}")
    
    # Build command (no JSON output file needed)
    cmd = [
        'python', 'vllm_inference.py',
        prompt_data['prompt'],
        '--model', model,
        '--prompter-api-key', settings['api_key'],
        '--index', settings['index_dir']
    ]
    
    # Get model-specific parameters
    if isinstance(settings['models'], dict) and model in settings['models']:
        params = settings['models'][model]
    else:
        # Fallback for command line override models
        params = getattr(settings, 'default_params', {})
    
    # Add optional parameters
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
    
    try:
        # Run the command and capture output
        print(f"🚀 Executing command...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            # Extract the response from stdout
            # The response is printed after "RESPONSE:" and before the next section
            stdout = result.stdout
            response_start = stdout.find("RESPONSE:")
            if response_start != -1:
                response_section = stdout[response_start + len("RESPONSE:"):].strip()
                # Find the end of the response (before any additional output like timing info)
                response_lines = response_section.split('\n')
                # Filter out empty lines and separator lines
                response_text = []
                for line in response_lines:
                    if line.strip() and not line.startswith('=') and not line.startswith('💾'):
                        response_text.append(line.strip())
                
                model_response = '\n'.join(response_text).strip()
                print(f"✅ Success: Got response ({len(model_response)} chars)")
                return True, model_response, params
            else:
                print(f"⚠️  Warning: Could not extract response from output")
                return False, "Could not extract response", params
        else:
            print(f"❌ Error: {result.stderr}")
            if result.stdout:
                print(f"📄 Output: {result.stdout}")
            return False, f"Error: {result.stderr}", params
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout: Command took longer than 5 minutes")
        return False, "Timeout after 5 minutes", {}
    except Exception as e:
        print(f"💥 Exception: {e}")
        return False, f"Exception: {str(e)}", {}

def main():
    parser = argparse.ArgumentParser(description='Sequential batch testing for vLLM inference')
    parser.add_argument('config_file', help='JSON configuration file')
    parser.add_argument('--models', nargs='+', help='Override models from config')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be run without executing')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_file)
    settings = config['settings']
    prompts = config['prompts']
    
    # Use all prompts
    if not prompts:
        print("No prompts found in configuration.")
        return
    
    # Use models from args or config
    if args.models:
        models = args.models
        # For command line models, use default params from first model in config
        default_params = list(settings['models'].values())[0] if settings['models'] else {}
    else:
        models = list(settings['models'].keys())
    
    # Create output directory
    output_dir = settings.get('output_dir', 'results')
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"📋 Found {len(prompts)} prompts to test")
    print(f"🤖 Testing with {len(models)} models: {', '.join(models)}")
    print(f"📁 Output directory: {output_dir}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - Commands that would be executed:")
        for prompt_data in prompts:
            for model in models:
                print(f"  - {prompt_data['id']} × {model}: {prompt_data['prompt'][:50]}...")
        return
    
    # Run tests
    total_tests = len(prompts) * len(models)
    current_test = 0
    successful_tests = 0
    
    results_summary = []
    
    for prompt_data in prompts:
        for model in models:
            current_test += 1
            print(f"\n📊 Progress: {current_test}/{total_tests}")
            
            success, response, params = run_single_prompt(prompt_data, model, settings, output_dir)
            
            results_summary.append({
                'prompt_id': prompt_data['id'],
                'prompt': prompt_data['prompt'],
                'model': model,
                'success': success,
                'response': response,
                'parameters': params,
                'timestamp': datetime.now().isoformat()
            })
            
            if success:
                successful_tests += 1
    
    # Save summary with all responses
    summary_file = f"{output_dir}/batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'config_file': args.config_file,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'run_timestamp': datetime.now().isoformat(),
            'results': results_summary
        }, f, indent=2)
    
    print(f"\n🎉 Batch testing complete!")
    print(f"✅ Successful: {successful_tests}/{total_tests}")
    print(f"📄 Summary saved to: {summary_file}")

if __name__ == '__main__':
    main()
