"""
Test Script for P2 Features - Advanced Generation

This script demonstrates and tests the P2 features:
1. Bug Injection Tasks
2. Enhanced Negative/Abstention Examples
3. Multi-Chunk Paper QA
4. Selective Questioning

Run this script to validate that all P2 components work correctly.
"""

import os
import sys
import logging
from pathlib import Path

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test code sample
SAMPLE_CODE = '''
def binary_search(arr, target):
    """
    Perform binary search on a sorted array.
    
    Args:
        arr: Sorted array to search in
        target: Value to search for
        
    Returns:
        Index of target if found, -1 otherwise
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


class SearchManager:
    """Manages different search algorithms."""
    
    def __init__(self, algorithm="binary"):
        self.algorithm = algorithm
        self.searches_performed = 0
    
    def search(self, data, target):
        """Perform search using configured algorithm."""
        self.searches_performed += 1
        
        if self.algorithm == "binary":
            return binary_search(data, target)
        else:
            return -1
    
    def get_stats(self):
        """Get search statistics."""
        return {"searches": self.searches_performed}
'''

# Sample paper chunks
SAMPLE_PAPER_CHUNKS = [
    """
    Abstract
    
    This paper presents a novel approach to machine learning optimization using 
    gradient-based methods. We propose a new algorithm that improves convergence 
    rates by 23% compared to existing methods. Our approach combines adaptive 
    learning rates with momentum-based updates to achieve better performance 
    on benchmark datasets.
    """,
    
    """
    Introduction
    
    Machine learning optimization has been a central challenge in the field 
    for decades. Traditional gradient descent methods often suffer from slow 
    convergence and can get trapped in local minima. Recent advances in 
    adaptive optimization have shown promise, but there remains room for 
    improvement in convergence speed and stability.
    """,
    
    """
    Methodology
    
    Our proposed algorithm, AdaptiveMomentum, combines several key innovations:
    1. Dynamic learning rate adjustment based on gradient variance
    2. Momentum coefficients that adapt to the optimization landscape
    3. Regularization terms that prevent overfitting
    
    The algorithm maintains a running average of gradients and their second moments,
    similar to Adam, but introduces novel adaptive mechanisms.
    """,
    
    """
    Results
    
    We evaluated our method on five benchmark datasets: MNIST, CIFAR-10, 
    ImageNet, Penn Treebank, and CoNLL-2003. Our algorithm achieved:
    - 23% faster convergence than Adam
    - 15% better final accuracy on image classification tasks  
    - 18% improvement in convergence stability
    
    Statistical significance tests (p < 0.01) confirm these improvements.
    """
]


def test_bug_injection():
    """Test P2.1: Bug Injection Tasks"""
    print("\n" + "="*50)
    print("Testing P2.1: Bug Injection Tasks")
    print("="*50)
    
    try:
        from data_generation.data_gen.tasks import BugInjector, create_bug_injector
        from data_generation.data_gen.symbols import SymbolExtractor, create_extraction_config
        
        # Create extractor for symbols with lenient token budget
        config = create_extraction_config(
            complexity_level="all", 
            max_symbols=5,
            token_range=(10, 1000),  # Very lenient for test - allow small functions
            min_lines_of_code=3,     # Allow smaller functions
            max_lines_of_code=200    # Reasonable upper limit
        )
        extractor = SymbolExtractor(config)
        
        # Create a test file
        test_file = Path("temp_test_code.py")
        test_file.write_text(SAMPLE_CODE)
        
        try:
            # Extract symbols
            symbols = extractor.extract_from_file(str(test_file))
            print(f"🔍 Found {len(symbols)} symbols to test with")
            
            if not symbols:
                print("❌ No symbols extracted for bug injection")
                return False
            
            # Create bug injector
            bug_injector = create_bug_injector(difficulty_level="mixed")
            
            test_symbol = symbols[0]  # Use first symbol
            print(f"🔧 Testing bug injection on: {test_symbol.name}")
            print(f"   Symbol type: {test_symbol.symbol_type.value}")
            print(f"   Token count: {test_symbol.token_count}")
            print(f"   Lines of code: {test_symbol.complexity.lines_of_code}")
            
            # Inject different types of bugs
            bug_types = ["off_by_one", "logic_error", "variable_name", "type_mismatch"]
            successful_injections = 0
            
            for bug_type_str in bug_types:
                try:
                    from data_generation.data_gen.tasks.bug_injector import BugType
                    bug_type = BugType(bug_type_str)
                    injection = bug_injector.inject_bug(test_symbol, bug_type)
                    
                    if injection:
                        successful_injections += 1
                        print(f"  ✅ {bug_type_str}: {injection.bug_description}")
                        print(f"     Severity: {injection.severity}")
                        print(f"     Fix: {injection.fix_explanation[:50]}...")
                    else:
                        print(f"  ⚠️ {bug_type_str}: Injection failed")
                        
                except Exception as e:
                    print(f"  ❌ {bug_type_str}: Error - {e}")
            
            # Get statistics
            stats = bug_injector.get_stats()
            print(f"\n📊 Bug Injection Stats:")
            print(f"  Total attempts: {stats.total_attempts}")
            print(f"  Successful: {stats.successful_injections}")
            print(f"  By type: {stats.bugs_by_type}")
            
            print(f"✅ Bug injection test completed ({successful_injections}/{len(bug_types)} successful)")
            return successful_injections > 0
            
        finally:
            if test_file.exists():
                test_file.unlink()
                
    except Exception as e:
        print(f"❌ Bug injection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_debug_generation():
    """Test P2.1: Debug Task Generation"""
    print("\n" + "="*50)
    print("Testing P2.1: Debug Task Generation")
    print("="*50)
    
    try:
        from data_generation.data_gen.tasks import create_debug_generator
        from data_generation.data_gen.assembly import get_config_manager
        from data_generation.data_gen.symbols import SymbolExtractor, create_extraction_config
        
        # Get components
        config_manager = get_config_manager()
        if not config_manager.config:
            print("⚠️ Configuration not loaded, using mock")
            return False
        
        # Mock LLM client
        class MockLLMClient:
            def messages_create(self, **kwargs):
                return type('obj', (object,), {
                    'content': [type('obj', (object,), {
                        'text': 'Mock debug explanation: This bug could cause IndexError.'
                    })]
                })
        
        mock_client = MockLLMClient()
        debug_generator = create_debug_generator(config_manager, mock_client)
        
        # Create test symbol with lenient token budget
        config = create_extraction_config(
            complexity_level="all", 
            max_symbols=5,
            token_range=(10, 1000),  # Very lenient for test
            min_lines_of_code=3,     # Allow smaller functions
            max_lines_of_code=200    # Reasonable upper limit
        )
        extractor = SymbolExtractor(config)
        
        test_file = Path("temp_test_code.py")
        test_file.write_text(SAMPLE_CODE)
        
        try:
            symbols = extractor.extract_from_file(str(test_file))
            print(f"🔍 Found {len(symbols)} symbols for debug generation")
            
            if not symbols:
                print("❌ No symbols for debug generation")
                return False
            
            test_symbol = symbols[0]
            print(f"🐛 Generating debug tasks for: {test_symbol.name}")
            print(f"   Token count: {test_symbol.token_count}")
            
            # Generate debug tasks
            debug_tasks = debug_generator.generate_debug_tasks(test_symbol, "moderate")
            
            print(f"✅ Generated {len(debug_tasks)} debug tasks")
            
            for i, task in enumerate(debug_tasks[:3]):  # Show first 3
                print(f"\n  Debug Task {i+1}:")
                print(f"    Type: {task.task_type}")
                print(f"    Question: {task.question[:80]}...")
                if hasattr(task, 'bug_injection') and task.bug_injection:
                    print(f"    Bug Type: {task.bug_injection.bug_type.value}")
                print(f"    Difficulty: {task.difficulty_level}")
            
            # Get statistics
            stats = debug_generator.get_stats()
            print(f"\n📊 Debug Generation Stats:")
            print(f"  Total tasks: {stats.total_tasks_generated}")
            print(f"  By type: {stats.tasks_by_type}")
            print(f"  By bug type: {stats.tasks_by_bug_type}")
            
            return len(debug_tasks) > 0
            
        finally:
            if test_file.exists():
                test_file.unlink()
                
    except Exception as e:
        print(f"❌ Debug generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_negative_examples():
    """Test P2.2: Enhanced Negative Examples"""
    print("\n" + "="*50)
    print("Testing P2.2: Enhanced Negative Examples")
    print("="*50)
    
    try:
        from data_generation.data_gen.tasks import create_negative_generator
        from data_generation.data_gen.assembly import get_config_manager
        from data_generation.data_gen.symbols import SymbolExtractor, create_extraction_config
        
        config_manager = get_config_manager()
        if not config_manager.config:
            print("⚠️ Configuration not loaded, using fallback")
            return False
        
        negative_generator = create_negative_generator(config_manager)
        
        # Create test symbol with lenient token budget
        config = create_extraction_config(
            complexity_level="all", 
            max_symbols=5,
            token_range=(10, 1000),  # Very lenient for test
            min_lines_of_code=3,     # Allow smaller functions
            max_lines_of_code=200    # Reasonable upper limit
        )
        extractor = SymbolExtractor(config)
        
        test_file = Path("temp_test_code.py")
        test_file.write_text(SAMPLE_CODE)
        
        try:
            symbols = extractor.extract_from_file(str(test_file))
            print(f"🔍 Found {len(symbols)} symbols for negative examples")
            
            if not symbols:
                print("❌ No symbols for negative example generation")
                return False
            
            test_symbol = symbols[0]
            print(f"🚫 Generating negative examples for: {test_symbol.name}")
            print(f"   Symbol type: {test_symbol.symbol_type.value}")
            
            # Generate negative examples
            negative_examples = negative_generator.generate_negative_examples(
                test_symbol, "moderate", count=5
            )
            
            print(f"✅ Generated {len(negative_examples)} negative examples")
            
            for i, example in enumerate(negative_examples[:3]):  # Show first 3
                print(f"\n  Negative Example {i+1}:")
                print(f"    Type: {example.negative_type.value}")
                print(f"    Question: {example.question[:80]}...")
                print(f"    Expected: {example.expected_response}")
                print(f"    Trap indicators: {len(example.trap_indicators)}")
                print(f"    Abstention cues: {len(example.abstention_cues)}")
            
            # Test adversarial examples
            adversarial = negative_generator.generate_adversarial_examples(test_symbol, "moderate")
            print(f"\n🎯 Generated {len(adversarial)} adversarial examples")
            
            # Get statistics
            stats = negative_generator.get_stats()
            print(f"\n📊 Negative Generation Stats:")
            print(f"  Total generated: {stats.total_generated}")
            print(f"  By type: {stats.by_type}")
            print(f"  Average trap indicators: {stats.average_trap_indicators:.1f}")
            
            return len(negative_examples) > 0
            
        finally:
            if test_file.exists():
                test_file.unlink()
                
    except Exception as e:
        print(f"❌ Enhanced negative examples test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_chunk_paper_qa():
    """Test P2.3: Multi-Chunk Paper QA"""
    print("\n" + "="*50)
    print("Testing P2.3: Multi-Chunk Paper QA")
    print("="*50)
    
    try:
        from data_generation.data_gen.tasks import create_paper_qa_generator
        from data_generation.data_gen.assembly import get_config_manager
        
        config_manager = get_config_manager()
        
        # Mock LLM client
        class MockLLMClient:
            def messages_create(self, **kwargs):
                user_content = kwargs.get('messages', [{}])[-1].get('content', '')
                if 'integrative' in user_content.lower():
                    response = "How do the proposed methods in the methodology relate to the experimental results?"
                else:
                    response = "The key finding is that the proposed algorithm shows significant improvement."
                
                return type('obj', (object,), {
                    'content': [type('obj', (object,), {'text': response})]
                })
        
        mock_client = MockLLMClient()
        paper_qa_generator = create_paper_qa_generator(config_manager, mock_client)
        
        print("📄 Testing multi-chunk paper QA generation...")
        
        # Classify chunks
        paper_chunks = paper_qa_generator.classify_chunks(SAMPLE_PAPER_CHUNKS)
        
        print(f"✅ Classified {len(paper_chunks)} paper chunks")
        for chunk in paper_chunks:
            print(f"  - {chunk.section_title}: {chunk.section_type.value}")
        
        # Generate integrative QA pairs
        print("\n🔗 Generating integrative QA pairs...")
        qa_pairs = paper_qa_generator.generate_integrative_qa_pairs(paper_chunks, count=3)
        
        print(f"✅ Generated {len(qa_pairs)} integrative QA pairs")
        
        for i, qa in enumerate(qa_pairs):
            print(f"\n  QA Pair {i+1}:")
            print(f"    Integration Type: {qa.integration_type}")
            print(f"    Question: {qa.question[:80]}...")
            print(f"    Chunks Used: {len(qa.source_chunks)}")
            print(f"    Sections: {qa.section_types_involved}")
            print(f"    Cross-reference: {qa.requires_cross_reference}")
        
        return len(qa_pairs) > 0
        
    except Exception as e:
        print(f"❌ Multi-chunk paper QA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_selective_questioning():
    """Test P2.4: Selective Questioning"""
    print("\n" + "="*50)
    print("Testing P2.4: Selective Questioning")
    print("="*50)
    
    try:
        from data_generation.data_gen.tasks import create_selective_questioner
        from data_generation.data_gen.assembly import get_config_manager
        from data_generation.data_gen.symbols import SymbolExtractor, create_extraction_config
        
        config_manager = get_config_manager()
        selective_questioner = create_selective_questioner(config_manager)
        
        # Test with code symbols
        config = create_extraction_config(
            complexity_level="all", 
            max_symbols=5,
            token_range=(10, 1000),  # Very lenient for test
            min_lines_of_code=3,     # Allow smaller functions
            max_lines_of_code=200    # Reasonable upper limit
        )
        extractor = SymbolExtractor(config)
        
        test_file = Path("temp_test_code.py")
        test_file.write_text(SAMPLE_CODE)
        
        try:
            symbols = extractor.extract_from_file(str(test_file))
            print(f"🔍 Found {len(symbols)} symbols for selective questioning")
            
            if not symbols:
                print("❌ No symbols for selective questioning")
                return False
            
            test_symbol = symbols[0]
            print(f"🎯 Testing selective questioning for: {test_symbol.name}")
            
            # Select questions for code symbol
            selected_templates = selective_questioner.select_questions_for_code_symbol(
                test_symbol, "moderate", count=4
            )
            
            print(f"✅ Selected {len(selected_templates)} question templates for code")
            
            for i, template in enumerate(selected_templates):
                print(f"\n  Template {i+1}:")
                print(f"    Question: {template.template}")
                print(f"    Category: {template.category.value}")
                print(f"    Difficulty: {template.difficulty}")
                print(f"    Focus Areas: {template.focus_areas}")
                print(f"    Weight: {template.weight:.2f}")
            
            # Test with paper chunks
            from data_generation.data_gen.tasks.paper_qa_generator import PaperChunk, SectionType
            
            sample_chunk = PaperChunk(
                content=SAMPLE_PAPER_CHUNKS[0],
                section_type=SectionType.ABSTRACT,
                section_title="Abstract",
                chunk_index=0
            )
            
            paper_templates = selective_questioner.select_questions_for_paper_chunk(
                sample_chunk, count=3
            )
            
            print(f"\n✅ Selected {len(paper_templates)} question templates for paper")
            
            for i, template in enumerate(paper_templates):
                print(f"\n  Paper Template {i+1}:")
                print(f"    Question: {template.template}")
                print(f"    Category: {template.category.value}")
                print(f"    Required Elements: {template.required_elements}")
            
            # Test adaptive question set
            adaptive_questions = selective_questioner.generate_adaptive_question_set(
                SAMPLE_CODE, "code", "moderate", focus_areas=["functionality", "algorithm"]
            )
            
            print(f"\n🔄 Generated {len(adaptive_questions)} adaptive questions")
            
            # Get statistics
            stats = selective_questioner.get_stats()
            print(f"\n📊 Selective Questioning Stats:")
            print(f"  Questions generated: {stats.questions_generated}")
            print(f"  By category: {stats.by_category}")
            print(f"  By difficulty: {stats.by_difficulty}")
            print(f"  Average relevance: {stats.average_relevance_score:.2f}")
            
            return len(selected_templates) > 0 and len(paper_templates) > 0
            
        finally:
            if test_file.exists():
                test_file.unlink()
                
    except Exception as e:
        print(f"❌ Selective questioning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all P2 feature tests."""
    print("🚀 Testing P2 Features - Advanced Generation")
    print("=" * 70)
    
    # Test each P2 component
    bug_injection_passed = test_bug_injection()
    debug_generation_passed = test_debug_generation()
    negative_examples_passed = test_enhanced_negative_examples()
    paper_qa_passed = test_multi_chunk_paper_qa()
    selective_questioning_passed = test_selective_questioning()
    
    # Summary
    print("\n" + "="*70)
    print("🎯 P2 Feature Test Summary")
    print("="*70)
    
    tests_passed = 0
    total_tests = 5
    
    if bug_injection_passed:
        print("✅ P2.1 Bug Injection Tasks: PASSED")
        tests_passed += 1
    else:
        print("❌ P2.1 Bug Injection Tasks: FAILED")
    
    if debug_generation_passed:
        print("✅ P2.1 Debug Generation: PASSED")
        tests_passed += 1
    else:
        print("❌ P2.1 Debug Generation: FAILED")
    
    if negative_examples_passed:
        print("✅ P2.2 Enhanced Negative Examples: PASSED")
        tests_passed += 1
    else:
        print("❌ P2.2 Enhanced Negative Examples: FAILED")
    
    if paper_qa_passed:
        print("✅ P2.3 Multi-Chunk Paper QA: PASSED")
        tests_passed += 1
    else:
        print("❌ P2.3 Multi-Chunk Paper QA: FAILED")
    
    if selective_questioning_passed:
        print("✅ P2.4 Selective Questioning: PASSED")
        tests_passed += 1
    else:
        print("❌ P2.4 Selective Questioning: FAILED")
    
    print(f"\n📊 Overall: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All P2 features working correctly!")
        print("\n💡 P2 Advanced Generation features ready for production use")
        print("   - Sophisticated bug injection and debugging tasks")
        print("   - Enhanced negative examples for abstention training")
        print("   - Multi-chunk integrative paper QA generation")
        print("   - Intelligent selective questioning based on content type")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("💡 Common fixes:")
        print("   - Ensure token budget allows small functions (min_tokens=10)")
        print("   - Check that symbol extraction is working properly")
        print("   - Verify configuration files are loaded correctly")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 