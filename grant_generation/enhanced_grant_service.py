#!/usr/bin/env python3
"""
enhanced_grant_service.py

Advanced grant generation service implementing a two-stage approach:
1. Information extraction from uploaded documents
2. Section generation using RAG + extracted info + overview
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
import re
from enhanced_document_processor import ProcessedDocument, EnhancedDocumentProcessor
import rag_service
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

@dataclass
class SectionContent:
    """Container for section generation results"""
    section_name: str
    generated_text: str
    extracted_info: str
    rag_context: str
    quality_score: float
    suggestions: List[str]

class EnhancedGrantService:
    def __init__(self):
        self.document_processor = EnhancedDocumentProcessor()
        
        # Enhanced section prompts with better structure
        self.section_prompts = {
            "Background": {
                "extraction_prompt": """Based on the grant overview and uploaded documents, extract ALL information relevant to the BACKGROUND section of a grant proposal. Focus on:

1. Current state of knowledge in the field
2. Previous research findings and literature
3. Existing gaps or problems in the field
4. Context that justifies this research

Grant Overview: {overview}

Extracted Information:""",

                "generation_prompt": """You are an expert grant writer. Write a comprehensive Background section using:

1. Grant Overview: {overview}
2. Extracted Information: {extracted_info}
3. Lab Context from RAG: {rag_context}

Write a scholarly Background section that:
- Establishes broad context in the field
- Identifies specific knowledge gaps
- Justifies the importance of addressing these gaps
- Uses formal academic tone
- Integrates information seamlessly from all sources

Background Section:"""
            },

            "Objectives": {
                "extraction_prompt": """Based on the grant overview and uploaded documents, extract ALL information relevant to the OBJECTIVES section. Focus on:

1. Main goals and aims mentioned
2. What the research hopes to achieve
3. Specific targets or outcomes desired
4. Problem statements that need addressing

Grant Overview: {overview}

Extracted Information:""",

                "generation_prompt": """You are an expert grant writer. Write compelling Objectives using:

1. Grant Overview: {overview}
2. Extracted Information: {extracted_info}
3. Lab Context from RAG: {rag_context}

Write clear Objectives that:
- State 2-3 high-level goals
- Directly address the Background problem
- Are achievable and measurable
- Align with the lab's expertise
- Use strong, confident language

Objectives Section:"""
            },

            "Specific Aims": {
                "extraction_prompt": """Based on the grant overview and uploaded documents, extract information for SPECIFIC AIMS:

1. Specific research questions or hypotheses
2. Detailed aims or sub-goals
3. Testable objectives
4. Methodological approaches for each aim

Grant Overview: {overview}

Extracted Information:""",

                "generation_prompt": """You are an expert grant writer. Write detailed Specific Aims using:

1. Grant Overview: {overview}
2. Extracted Information: {extracted_info}
3. Lab Context from RAG: {rag_context}

Write 2-4 numbered Specific Aims that:
- Are concrete and testable
- Include brief approach descriptions
- Build logically toward the objectives
- Are feasible within project scope
- Reference lab capabilities when relevant

Specific Aims Section:"""
            },

            "Methods": {
                "extraction_prompt": """Extract methodological information for the METHODS section:

1. Experimental procedures and protocols
2. Data collection methods
3. Analysis techniques and tools
4. Equipment and technologies mentioned
5. Statistical or computational approaches

Grant Overview: {overview}

Extracted Information:""",

                "generation_prompt": """You are an expert grant writer. Write a detailed Methods section using:

1. Grant Overview: {overview}
2. Extracted Information: {extracted_info}
3. Lab Context from RAG: {rag_context}

Write comprehensive Methods that:
- Detail approach for each Specific Aim
- Specify data collection procedures
- Describe analysis methods
- Justify methodological choices
- Reference lab resources and expertise
- Include timeline considerations

Methods Section:"""
            },

            "Preliminary work": {
                "extraction_prompt": """Extract information for PRELIMINARY WORK section:

1. Previous results and pilot studies
2. Baseline or foundational work completed
3. Proof-of-concept data
4. Relevant prior publications or findings
5. Technical feasibility demonstrations

Grant Overview: {overview}

Extracted Information:""",

                "generation_prompt": """You are an expert grant writer. Write the Preliminary Work section using:

1. Grant Overview: {overview}
2. Extracted Information: {extracted_info}
3. Lab Context from RAG: {rag_context}

Write compelling Preliminary Work that:
- Presents relevant prior results
- Demonstrates feasibility
- Shows team's track record
- Connects to proposed research
- Builds confidence in success
- Uses specific data when available

Preliminary Work Section:"""
            }
        }
        
        # Add prompts for remaining sections...
        self._add_remaining_section_prompts()

    def _add_remaining_section_prompts(self):
        """Add prompts for remaining sections"""
        remaining_sections = {
            "Impact/Relevance": {
                "extraction_prompt": """Extract information for IMPACT/RELEVANCE:

1. Potential societal or scientific impact
2. Relevance to field or broader applications
3. Benefits to community or patients
4. Policy implications
5. Commercial or translational potential

Grant Overview: {overview}

Extracted Information:""",

                "generation_prompt": """Write the Impact/Relevance section that demonstrates the broader significance of this research and its potential to advance the field or benefit society."""
            },

            "Feasibility, Risks and Mitigation Strategies": {
                "extraction_prompt": """Extract information for FEASIBILITY and RISK ASSESSMENT:

1. Technical challenges and limitations
2. Resource requirements and availability
3. Potential risks to project success
4. Alternative approaches or backup plans
5. Timeline and logistical considerations

Grant Overview: {overview}

Extracted Information:""",

                "generation_prompt": """Write the Feasibility, Risks and Mitigation section that honestly addresses potential challenges while demonstrating preparedness and confidence in success."""
            }
            # ... continue for all sections
        }
        
        self.section_prompts.update(remaining_sections)

    def extract_section_info(self, processed_docs: List[ProcessedDocument], 
                           section_name: str, grant_overview: str) -> str:
        """Stage 1: Extract relevant information for a specific section"""
        try:
            if section_name not in self.section_prompts:
                return ""
            
            # Create context from processed documents
            context_parts = []
            for doc in processed_docs:
                if section_name in doc.sections and doc.sections[section_name].strip():
                    context_parts.append(f"From {doc.filename}:\n{doc.sections[section_name]}")
            
            document_context = "\n\n".join(context_parts)
            
            # Use extraction prompt
            extraction_prompt = self.section_prompts[section_name]["extraction_prompt"].format(
                overview=grant_overview
            )
            
            full_prompt = f"{extraction_prompt}\n\nDocument Content:\n{document_context}\n\nExtracted Information:"
            
            # Use LabGPT for extraction (without RAG)
            response = rag_service.generator(full_prompt, max_length=2000, num_return_sequences=1)
            extracted_info = response[0]["generated_text"].split("Extracted Information:")[-1].strip()
            
            return extracted_info
            
        except Exception as e:
            logging.error(f"Error extracting info for {section_name}: {e}")
            return ""

    def generate_section_content(self, section_name: str, grant_overview: str, 
                               extracted_info: str, use_rag: bool = True) -> SectionContent:
        """Stage 2: Generate section content using extracted info + RAG"""
        try:
            # Get RAG context if requested
            rag_context = ""
            if use_rag:
                # Create query for RAG based on section and overview
                rag_query = f"Information relevant to {section_name} section for: {grant_overview[:200]}..."
                rag_context = rag_service.get_rag_answer(rag_query).split("Answer:")[-1].strip()
            
            # Use generation prompt
            if section_name not in self.section_prompts:
                return SectionContent(section_name, "", extracted_info, rag_context, 0.0, ["Section not supported"])
            
            generation_prompt = self.section_prompts[section_name]["generation_prompt"].format(
                overview=grant_overview,
                extracted_info=extracted_info,
                rag_context=rag_context
            )
            
            # Generate section
            response = rag_service.generator(generation_prompt, max_length=3000, num_return_sequences=1)
            generated_text = response[0]["generated_text"].split("Section:")[-1].strip()
            
            # Quality assessment
            quality_score, suggestions = self.assess_section_quality(generated_text, section_name)
            
            return SectionContent(
                section_name=section_name,
                generated_text=generated_text,
                extracted_info=extracted_info,
                rag_context=rag_context,
                quality_score=quality_score,
                suggestions=suggestions
            )
            
        except Exception as e:
            logging.error(f"Error generating {section_name}: {e}")
            return SectionContent(section_name, "", extracted_info, "", 0.0, [f"Error: {str(e)}"])

    def assess_section_quality(self, text: str, section_name: str) -> Tuple[float, List[str]]:
        """Assess the quality of generated section content"""
        suggestions = []
        score = 5.0  # Start with perfect score
        
        # Basic checks
        if len(text.strip()) < 100:
            score -= 2.0
            suggestions.append("Section is too short - consider adding more detail")
        
        if len(text.strip()) > 2000:
            score -= 0.5
            suggestions.append("Section might be too long - consider condensing")
        
        # Section-specific checks
        if section_name == "Background":
            if "gap" not in text.lower() and "problem" not in text.lower():
                score -= 1.0
                suggestions.append("Consider explicitly identifying research gaps or problems")
                
        elif section_name == "Objectives":
            if not re.search(r'\b(aim|goal|objective)\b', text.lower()):
                score -= 1.0
                suggestions.append("Ensure objectives are clearly stated")
                
        elif section_name == "Specific Aims":
            if not re.search(r'(aim \d|aim one|aim two|aim three)', text.lower()):
                score -= 1.0
                suggestions.append("Consider numbering or clearly delineating specific aims")
        
        # Content quality checks
        sentences = text.split('. ')
        if len(sentences) < 3:
            score -= 1.0
            suggestions.append("Add more detailed explanations")
        
        # Ensure score is between 0 and 5
        score = max(0.0, min(5.0, score))
        
        if score >= 4.0:
            suggestions.append("High quality content - ready for review")
        elif score >= 3.0:
            suggestions.append("Good content with minor improvements needed")
        else:
            suggestions.append("Significant improvements recommended")
        
        return score, suggestions

    def refine_section_with_feedback(self, section_content: SectionContent, 
                                   user_feedback: str, grant_overview: str) -> SectionContent:
        """Refine section based on user feedback"""
        try:
            refinement_prompt = f"""You are an expert grant writer. The user has provided feedback on the {section_content.section_name} section. Please revise the content accordingly.

Grant Overview: {grant_overview}

Current Section Content:
{section_content.generated_text}

User Feedback: {user_feedback}

Previously Extracted Information: {section_content.extracted_info}

Lab Context: {section_content.rag_context}

Please provide a revised {section_content.section_name} section that addresses the user's feedback while maintaining high quality:

Revised Section:"""
            
            response = rag_service.generator(refinement_prompt, max_length=3000, num_return_sequences=1)
            refined_text = response[0]["generated_text"].split("Revised Section:")[-1].strip()
            
            # Assess quality of refined version
            quality_score, suggestions = self.assess_section_quality(refined_text, section_content.section_name)
            
            return SectionContent(
                section_name=section_content.section_name,
                generated_text=refined_text,
                extracted_info=section_content.extracted_info,
                rag_context=section_content.rag_context,
                quality_score=quality_score,
                suggestions=suggestions
            )
            
        except Exception as e:
            logging.error(f"Error refining {section_content.section_name}: {e}")
            return section_content

    def generate_complete_section(self, processed_docs: List[ProcessedDocument], 
                                section_name: str, grant_overview: str) -> SectionContent:
        """Complete pipeline: extract info then generate section"""
        # Stage 1: Extract relevant information
        extracted_info = self.extract_section_info(processed_docs, section_name, grant_overview)
        
        # Stage 2: Generate section content
        section_content = self.generate_section_content(
            section_name, grant_overview, extracted_info, use_rag=True
        )
        
        return section_content 