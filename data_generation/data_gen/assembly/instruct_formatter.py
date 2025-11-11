"""
Instruct Format Converter for Fine-tuning Data

This module provides conversion from the internal QA and task formats to the
standard instruction fine-tuning format used by most modern LLMs.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import json
import logging

LABGPT_SYSTEM_PROMPT = """You are LABGPT, an advanced AI assistant specialized in laboratory research, computational biology, and scientific programming. You were developed to assist researchers at the BHK Lab and similar research institutions.

Your core capabilities include:
- Analyzing and generating code in multiple languages (Python, R, C, C++) for scientific computing and bioinformatics
- Understanding and explaining research papers, methodologies, and scientific concepts
- Assisting with grant writing and research documentation
- Debugging scientific code and suggesting optimizations
- Providing expertise in computational biology, pharmacogenomics, and medical imaging

Key principles:
- Always provide accurate, precise, and helpful responses
- When you don't have sufficient information to answer a question, clearly state "I don't have enough information to answer that"
- Maintain scientific rigor and precision in all responses
- Provide code examples that follow best practices and are well-documented
- Consider computational efficiency and reproducibility in scientific workflows

You should be helpful, precise, and thorough while maintaining a professional tone appropriate for academic and research environments."""


@dataclass
class InstructMessage:
    """Single message in instruct format."""
    role: str  # "user", "assistant", or "system"
    content: str


@dataclass
class InstructExample:
    """Complete instruct example with messages."""
    messages: List[InstructMessage]
    metadata: Dict[str, Any] = field(default_factory=dict)


class InstructFormatter:
    """Converts QA pairs and tasks to instruct format."""
    
    def __init__(self, include_system_prompt: bool = True):
        """
        Initialize the instruct formatter.

        Args:
            include_system_prompt: Whether to include system prompts in examples
        """
        self.include_system_prompt = include_system_prompt
        self.system_prompt = LABGPT_SYSTEM_PROMPT
        self.logger = logging.getLogger(__name__)
    
    def format_qa_pair(self, qa_pair: Dict[str, Any], symbol_info: Dict[str, Any]) -> InstructExample:
        """
        Convert a QA pair to instruct format.
        
        Args:
            qa_pair: QA pair dictionary
            symbol_info: Symbol information for context
            
        Returns:
            InstructExample in proper format
        """
        messages = []
        
        if self.include_system_prompt:
            messages.append(InstructMessage(role="system", content=self.system_prompt))
        
        # Add user question
        user_content = self._format_user_content(qa_pair, symbol_info)
        messages.append(InstructMessage(role="user", content=user_content))
        
        # Add assistant response
        assistant_content = self._format_assistant_content(qa_pair)
        messages.append(InstructMessage(role="assistant", content=assistant_content))
        
        # Create metadata
        metadata = {
            "source_type": "qa_pair",
            "language": symbol_info.get("language", "unknown"),
            "symbol_name": symbol_info.get("symbol_name", ""),
            "symbol_type": symbol_info.get("symbol_type", ""),
            "file": symbol_info.get("file", ""),
            "complexity": qa_pair.get("complexity", "unknown"),
            "focus": qa_pair.get("focus", ""),
            "negative": qa_pair.get("negative", False),
            "confidence": qa_pair.get("confidence", 0.0),
            "citations": qa_pair.get("citations", [])
        }
        
        return InstructExample(messages=messages, metadata=metadata)
    
    def format_debug_task(self, debug_task: Dict[str, Any], symbol_info: Dict[str, Any]) -> InstructExample:
        """
        Convert a debug task to instruct format.
        
        Args:
            debug_task: Debug task dictionary
            symbol_info: Symbol information for context
            
        Returns:
            InstructExample in proper format
        """
        messages = []
        
        if self.include_system_prompt:
            messages.append(InstructMessage(role="system", content=self.system_prompt))
        
        # Add user question
        user_content = self._format_debug_user_content(debug_task, symbol_info)
        messages.append(InstructMessage(role="user", content=user_content))
        
        # Add assistant response
        assistant_content = debug_task.get("expected_answer", "I need to analyze this code to identify any bugs.")
        messages.append(InstructMessage(role="assistant", content=assistant_content))
        
        # Create metadata
        metadata = {
            "source_type": "debug_task",
            "language": symbol_info.get("language", "unknown"),
            "symbol_name": symbol_info.get("symbol_name", ""),
            "symbol_type": symbol_info.get("symbol_type", ""),
            "file": symbol_info.get("file", ""),
            "task_type": debug_task.get("task_type", ""),
            "bug_type": debug_task.get("bug_type", ""),
            "bug_location": debug_task.get("bug_location", ""),
            "severity": debug_task.get("severity", ""),
            "difficulty": debug_task.get("difficulty", "")
        }
        
        return InstructExample(messages=messages, metadata=metadata)
    
    def format_paper_qa(self, paper_qa: Dict[str, Any], paper_info: Dict[str, Any]) -> InstructExample:
        """
        Convert a paper QA to instruct format.
        
        Args:
            paper_qa: Paper QA dictionary
            paper_info: Paper information for context
            
        Returns:
            InstructExample in proper format
        """
        messages = []
        
        if self.include_system_prompt:
            messages.append(InstructMessage(role="system", content=self.system_prompt))
        
        # Add user question
        user_content = paper_qa.get("question", "")
        messages.append(InstructMessage(role="user", content=user_content))
        
        # Add assistant response
        assistant_content = paper_qa.get("answer", "")
        messages.append(InstructMessage(role="assistant", content=assistant_content))
        
        # Create metadata
        metadata = {
            "source_type": "paper_qa",
            "file": paper_info.get("file", ""),
            "chunk_indices": paper_qa.get("chunk_indices", []),
            "integration_type": paper_qa.get("integration_type", ""),
            "requires_cross_reference": paper_qa.get("requires_cross_reference", False)
        }
        
        return InstructExample(messages=messages, metadata=metadata)
    
    def _format_user_content(self, qa_pair: Dict[str, Any], symbol_info: Dict[str, Any]) -> str:
        """Format user content for code QA."""
        question = qa_pair.get("question", "")
        
        # Include code context if available
        if symbol_info.get("source_code"):
            source_code = symbol_info["source_code"]
            file_path = symbol_info.get("file", "")
            start_line = symbol_info.get("start_line", 1)
            
            user_content = f"""Here is the code from {file_path} (starting at line {start_line}):

```{symbol_info.get("language", "")}
{source_code}
```

{question}"""
        else:
            user_content = question
        
        return user_content
    
    def _format_debug_user_content(self, debug_task: Dict[str, Any], symbol_info: Dict[str, Any]) -> str:
        """Format user content for debug tasks."""
        question = debug_task.get("question", "")
        
        # Include code context if available
        if symbol_info.get("source_code"):
            source_code = symbol_info["source_code"]
            file_path = symbol_info.get("file", "")
            start_line = symbol_info.get("start_line", 1)
            
            user_content = f"""Please analyze this code from {file_path} (starting at line {start_line}) for bugs or issues:

```{symbol_info.get("language", "")}
{source_code}
```

{question}"""
        else:
            user_content = question
        
        return user_content
    
    def _format_assistant_content(self, qa_pair: Dict[str, Any]) -> str:
        """Format assistant content for QA pairs."""
        answer = qa_pair.get("answer", "")
        citations = qa_pair.get("citations", [])
        
        # Add citations if available
        if citations and not qa_pair.get("negative", False):
            citation_text = "\n\nReferences:\n" + "\n".join(f"- {cite}" for cite in citations)
            answer += citation_text
        
        return answer
    
    def convert_dataset_to_instruct(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert entire dataset to instruct format.
        
        Args:
            dataset: List of dataset entries
            
        Returns:
            List of instruct examples in JSON format
        """
        instruct_examples = []
        
        for entry in dataset:
            # Convert QA pairs
            for qa_pair in entry.get("qa_pairs", []):
                try:
                    instruct_example = self.format_qa_pair(qa_pair, entry)
                    instruct_examples.append(self._to_dict(instruct_example))
                except Exception as e:
                    self.logger.error(f"Error converting QA pair to instruct format: {e}")
            
            # Convert debug tasks
            for debug_task in entry.get("debugging_tasks", []):
                try:
                    instruct_example = self.format_debug_task(debug_task, entry)
                    instruct_examples.append(self._to_dict(instruct_example))
                except Exception as e:
                    self.logger.error(f"Error converting debug task to instruct format: {e}")
        
        return instruct_examples
    
    def convert_paper_dataset_to_instruct(self, paper_dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert paper dataset to instruct format.
        
        Args:
            paper_dataset: List of paper dataset entries
            
        Returns:
            List of instruct examples in JSON format
        """
        instruct_examples = []
        
        for entry in paper_dataset:
            for paper_qa in entry.get("qa_pairs", []):
                try:
                    instruct_example = self.format_paper_qa(paper_qa, entry)
                    instruct_examples.append(self._to_dict(instruct_example))
                except Exception as e:
                    self.logger.error(f"Error converting paper QA to instruct format: {e}")
        
        return instruct_examples
    
    def _to_dict(self, instruct_example: InstructExample) -> Dict[str, Any]:
        """Convert InstructExample to dictionary format."""
        return {
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in instruct_example.messages
            ],
            "metadata": instruct_example.metadata
        }
    
    def save_instruct_dataset(self, instruct_examples: List[Dict[str, Any]], output_path: str):
        """
        Save instruct examples to JSON file.
        
        Args:
            instruct_examples: List of instruct examples
            output_path: Output file path
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(instruct_examples, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(instruct_examples)} instruct examples to {output_path}")
    
    def save_instruct_jsonl(self, instruct_examples: List[Dict[str, Any]], output_path: str):
        """
        Save instruct examples to JSONL file (one example per line).
        
        Args:
            instruct_examples: List of instruct examples
            output_path: Output file path
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in instruct_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Saved {len(instruct_examples)} instruct examples to {output_path}")


def create_instruct_formatter(system_prompts: bool = True) -> InstructFormatter:
    """
    Create an instruct formatter with default settings.
    
    Args:
        system_prompts: Whether to include system prompts
        
    Returns:
        Configured InstructFormatter
    """
    return InstructFormatter(include_system_prompt=system_prompts)