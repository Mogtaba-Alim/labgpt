#!/usr/bin/env python3
"""
Test script to verify that the modified scripts use user-provided inputs correctly
"""

import os
import sys
import tempfile
import json

def test_generateCodeFineTune_with_user_repos():
    """Test that generateCodeFineTune uses user-provided repository URLs"""
    print("Testing generateCodeFineTune with user-provided repos...")
    
    # Import the module
    import generateCodeFineTune
    
    # Test with a small set of user-provided repos
    user_repos = [
        'https://github.com/octocat/Hello-World.git',  # Small test repo
    ]
    
    print(f"User-provided repositories: {user_repos}")
    
    # Test the new function
    try:
        result = generateCodeFineTune.create_dataset_from_user_repos(
            user_repos, 
            output_dir='test_datasets', 
            target_dir='test_repos'
        )
        print("✓ generateCodeFineTune.create_dataset_from_user_repos() works correctly")
        return True
    except Exception as e:
        print(f"✗ Error in generateCodeFineTune: {e}")
        return False

def test_createFinalDataOutput_with_user_papers_dir():
    """Test that createFinalDataOutput uses user-provided papers directory"""
    print("Testing createFinalDataOutput with user-provided papers directory...")
    
    # Import the module  
    import createFinalDataOutput
    
    # Create a temporary directory with a test PDF (placeholder)
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a dummy .txt file and convert it to a simple PDF for testing
        test_txt_path = os.path.join(temp_dir, "test_paper.txt")
        test_pdf_path = os.path.join(temp_dir, "test_paper.pdf")
        with open(test_txt_path, 'w') as f:
            f.write("dummy pdf content for testing")

        # Convert the .txt to a real PDF using reportlab
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            c = canvas.Canvas(test_pdf_path, pagesize=letter)
            c.drawString(100, 750, "dummy pdf content for testing")
            c.save()
        except ImportError:
            print("✗ reportlab is required to create a dummy PDF. Please install it with 'pip install reportlab'.")
            return False

        print(f"User-provided papers directory: {temp_dir}")
        
        # Test the new function with a real PDF file
        try:
            # Create dummy code datasets first
            dummy_code_train = [{"test": "data"}]
            dummy_code_val = [{"test": "data"}]
            
            with open("code_combined_train_dataset.json", "w") as f:
                json.dump(dummy_code_train, f)
            with open("code_combined_val_dataset.json", "w") as f:
                json.dump(dummy_code_val, f)
            
            # Test that the function accepts user-provided directory
            train_size, val_size = createFinalDataOutput.process_papers_and_combine_with_code(temp_dir)
            print("✓ createFinalDataOutput.process_papers_and_combine_with_code() works correctly")
            return True
        except Exception as e:
            print(f"✗ Error in createFinalDataOutput: {e}")
            return False
        finally:
            # Clean up dummy files
            for file in ["code_combined_train_dataset.json", "code_combined_val_dataset.json"]:
                if os.path.exists(file):
                    os.remove(file)
            # Clean up dummy files in temp_dir
            for file in [test_txt_path, test_pdf_path]:
                if os.path.exists(file):
                    os.remove(file)

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing User Input Modifications")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Repository URLs
    test_results.append(test_generateCodeFineTune_with_user_repos())
    
    print("-" * 40)
    
    # Test 2: Papers Directory
    test_results.append(test_createFinalDataOutput_with_user_papers_dir())
    
    print("=" * 60)
    print("Test Results Summary:")
    print(f"Tests passed: {sum(test_results)}/{len(test_results)}")
    
    if all(test_results):
        print("✓ All tests passed! User inputs are working correctly.")
    else:
        print("✗ Some tests failed. Check the implementation.")
    
    return all(test_results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 