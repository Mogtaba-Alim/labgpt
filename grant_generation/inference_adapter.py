"""
inference_adapter.py

Wraps inference.py functions for grant-specific use.
Ensures consistent parameters across all sections and proper multi-turn conversation handling.
"""

import time
import logging
from typing import List, Dict, Tuple, Optional
from inference import (
    build_messages,
    generate_messages,
    retrieve_relevant_chunks_rag,
    LABGPT_SYSTEM
)

logger = logging.getLogger(__name__)


class GrantInferenceAdapter:
    """
    Adapter for grant generation using inference.py pipeline.

    Features:
    - Consistent parameters for all sections (no per-section variation)
    - Multi-turn conversation support for refinements
    - Automatic context retrieval with RAGPipeline
    - Proper use of chat templates and messages format
    - Uses trained LabGPT model via inference.py

    Parameters (same for all sections, as specified):
    - Temperature: 0.4 (focused, less creative)
    - Max tokens: 800 (sufficient for most grant sections)
    - Top-k: 5 (retrieve 5 most relevant chunks)
    - Preset: "research" (query expansion + auto-k + reranking)
    """

    # Same parameters for ALL sections (per requirement #1)
    DEFAULT_TEMPERATURE = 0.4
    DEFAULT_MAX_TOKENS = 800
    DEFAULT_TOP_K = 5
    PRESET = "research"  # Full feature set (expansion, auto-k, reranking)

    def generate_section(
        self,
        section_query: str,
        project_index_dir: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> Tuple[str, List[Dict], float]:
        """
        Generate section content using inference.py pipeline.

        Process:
        1. Retrieve context from project RAG index with citations
        2. Build messages array (system + user with context)
        3. Generate using trained LabGPT model with consistent parameters
        4. Return content, citations, and timing

        Args:
            section_query: Query describing what to generate
            project_index_dir: Path to project RAG index
            conversation_history: Previous messages for multi-turn (optional)

        Returns:
            Tuple of (generated_text, citation_objects, generation_time_sec)
        """
        start_time = time.time()

        logger.info(f"Generating section with query: {section_query[:100]}...")

        try:
            # Retrieve context with RAGPipeline (hybrid + reranking + expansion)
            context_items = retrieve_relevant_chunks_rag(
                query=section_query,
                index_dir=project_index_dir,
                top_k=self.DEFAULT_TOP_K,
                expand=True,        # PRF query expansion
                cited_spans=True,   # Extract cited spans for highlighting
                preset=self.PRESET  # Use research preset
            )

            logger.info(f"Retrieved {len(context_items)} context chunks")

            # Build messages
            if conversation_history:
                # Multi-turn: append to existing conversation
                messages = conversation_history + [{
                    "role": "user",
                    "content": section_query
                }]
            else:
                # First turn: use build_messages from inference.py
                # This adds system prompt + user message with context
                messages = build_messages(section_query, context_items)

            # Generate with consistent parameters
            response = generate_messages(
                messages=messages,
                max_new_tokens=self.DEFAULT_MAX_TOKENS,
                temperature=self.DEFAULT_TEMPERATURE
            )

            generation_time = time.time() - start_time

            logger.info(f"Generation complete in {generation_time:.2f}s")

            return response, context_items, generation_time

        except Exception as e:
            logger.error(f"Error during section generation: {e}")
            raise

    def refine_section(
        self,
        feedback: str,
        conversation_history: List[Dict],
        project_index_dir: str,
        retrieve_new_context: bool = False
    ) -> Tuple[str, List[Dict], float]:
        """
        Refine section based on user feedback (multi-turn conversation).

        Process:
        1. Append feedback to conversation history
        2. Optionally retrieve new context if feedback suggests it
        3. Generate refinement maintaining conversation context
        4. Return refined content and any new citations

        Args:
            feedback: User's refinement request/feedback
            conversation_history: Previous messages in conversation
            project_index_dir: Path to project RAG index
            retrieve_new_context: Force new context retrieval

        Returns:
            Tuple of (refined_text, new_citations, generation_time_sec)
        """
        start_time = time.time()

        logger.info(f"Refining section with feedback: {feedback[:100]}...")

        try:
            # Append feedback to conversation
            messages = conversation_history + [{
                "role": "user",
                "content": f"Please refine the section based on this feedback:\n\n{feedback}"
            }]

            # Check if we need new context
            need_context = retrieve_new_context or self._needs_more_context(feedback)

            new_citations = []
            if need_context:
                # Retrieve additional context based on feedback
                context_items = retrieve_relevant_chunks_rag(
                    query=feedback,
                    index_dir=project_index_dir,
                    top_k=3,  # Fewer chunks for refinement
                    expand=True,
                    cited_spans=True,
                    preset=self.PRESET
                )

                new_citations = context_items

                logger.info(f"Retrieved {len(context_items)} new context chunks for refinement")

                # Add new context to messages
                if context_items:
                    context_text = "\n\n".join([
                        f"[{i+1}] {item.get('citation', 'Unknown source')}\n{item.get('text', '')[:300]}..."
                        for i, item in enumerate(context_items)
                    ])

                    messages.append({
                        "role": "user",
                        "content": f"Additional context that may help:\n{context_text}"
                    })

            # Generate refinement
            response = generate_messages(
                messages=messages,
                max_new_tokens=self.DEFAULT_MAX_TOKENS,
                temperature=self.DEFAULT_TEMPERATURE
            )

            generation_time = time.time() - start_time

            logger.info(f"Refinement complete in {generation_time:.2f}s")

            return response, new_citations, generation_time

        except Exception as e:
            logger.error(f"Error during section refinement: {e}")
            raise

    @staticmethod
    def _needs_more_context(feedback: str) -> bool:
        """
        Heuristic to determine if feedback suggests need for new information.

        Keywords that indicate user wants more content or different information.

        Args:
            feedback: User feedback text

        Returns:
            True if feedback suggests need for new context
        """
        # Keywords suggesting need for additional information
        expansion_keywords = [
            'add', 'include', 'more', 'explain', 'expand',
            'detail', 'elaborate', 'cite', 'reference',
            'discuss', 'describe', 'cover', 'address'
        ]

        feedback_lower = feedback.lower()
        return any(keyword in feedback_lower for keyword in expansion_keywords)

    @staticmethod
    def _format_context(context_items: List[Dict]) -> str:
        """
        Format context items into readable text block.

        Args:
            context_items: List of citation dicts

        Returns:
            Formatted context string
        """
        context_parts = []
        for i, item in enumerate(context_items, 1):
            citation = item.get('citation', f"Source {i}")
            text = item.get('text', '')
            # Truncate long texts
            text_excerpt = text[:300] + "..." if len(text) > 300 else text
            context_parts.append(f"[{i}] {citation}\n{text_excerpt}")

        return "\n\n".join(context_parts)

    def get_parameters_info(self) -> Dict:
        """
        Get information about the generation parameters used.

        Returns:
            Dict with parameter values
        """
        return {
            'temperature': self.DEFAULT_TEMPERATURE,
            'max_tokens': self.DEFAULT_MAX_TOKENS,
            'top_k': self.DEFAULT_TOP_K,
            'preset': self.PRESET,
            'note': 'Same parameters for all sections'
        }
