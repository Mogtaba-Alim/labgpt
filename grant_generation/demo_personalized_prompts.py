#!/usr/bin/env python3
"""
Demo script to show the personalized prompts for each grant section
"""

import sys
sys.path.append('.')
from app import SECTIONS_ORDER, SECTION_PROMPTS

def demo_personalized_prompts():
    """Demonstrate the personalized prompts for each section"""
    print("=== GRANT GENERATION: PERSONALIZED PROMPTS DEMO ===\n")
    print(f"Total sections supported: {len(SECTIONS_ORDER)}")
    print(f"Total personalized prompts: {len(SECTION_PROMPTS)}")
    
    # Check if all sections have prompts
    missing_prompts = set(SECTIONS_ORDER) - set(SECTION_PROMPTS.keys())
    if missing_prompts:
        print(f"\n⚠️  WARNING: Missing prompts for: {missing_prompts}")
    else:
        print("\n✅ All sections have personalized prompts!")
    
    print("\n" + "="*80)
    print("SECTION PROMPTS OVERVIEW:")
    print("="*80)
    
    for i, section in enumerate(SECTIONS_ORDER, 1):
        print(f"\n{i:2d}. {section.upper()}")
        print("-" * (len(section) + 4))
        
        prompt = SECTION_PROMPTS.get(section, "No personalized prompt available")
        
        # Show first 200 characters of the prompt
        if len(prompt) > 200:
            preview = prompt[:200] + "..."
        else:
            preview = prompt
            
        print(f"Prompt preview: {preview}")
        print(f"Full prompt length: {len(prompt)} characters")
    
    print("\n" + "="*80)
    print("EXAMPLE: Full prompt for 'Background' section:")
    print("="*80)
    background_prompt = SECTION_PROMPTS.get("Background", "No prompt found")
    print(background_prompt)
    
    return True

if __name__ == "__main__":
    demo_personalized_prompts() 