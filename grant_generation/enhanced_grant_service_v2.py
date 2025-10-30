"""
enhanced_grant_service_v2.py

Modern grant generation service using:
- RAG/pipeline.py for hybrid retrieval
- inference.py message format with trained LabGPT model
- Consistent parameters across all sections
- Multi-turn conversation support for refinements
- Structured citations in MLA format

Replaces enhanced_grant_service.py which used legacy rag_service.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .corpus_manager import GrantCorpusManager
from .database import GrantDatabase
from .inference_adapter import GrantInferenceAdapter

logger = logging.getLogger(__name__)


@dataclass
class SectionResult:
    """Container for section generation/refinement results."""
    draft_id: str
    section_name: str
    content: str
    citations: List[Dict]
    quality_score: float
    quality_feedback: str
    version: int
    generation_time: float


class EnhancedGrantService:
    """
    Modern grant generation service.

    Features:
    - Uses RAGPipeline for hybrid retrieval with citations
    - Uses inference.py message format with trained LabGPT
    - Consistent parameters: temp=0.4, max_tokens=800, top_k=5
    - Multi-turn refinement with conversation history
    - Quality assessment with citation checks
    - SQLite storage for versioning and lineage
    """

    # Section-specific query templates
    # These guide what information to retrieve and generate
    # Enhanced with detailed instructions for high-quality grant writing
    SECTION_PROMPTS = {
        "Background": """You are an expert grant writer preparing the BACKGROUND section for a research proposal.

Grant Topic: {overview}

Write a comprehensive Background section that establishes the scientific foundation for this research. Your writing should:

CONTENT REQUIREMENTS:
1. Current State of Knowledge
   - Establish the broad context and significance of the research area
   - Summarize key findings from relevant literature
   - Identify major advances and breakthroughs in the field
   - Position this work within the larger scientific landscape

2. Knowledge Gaps and Problems
   - Explicitly identify specific gaps in current understanding
   - Highlight limitations of existing approaches or technologies
   - Describe unmet needs or unresolved questions
   - Explain why these gaps are important and need addressing

3. Research Justification
   - Connect the identified gaps to the proposed research
   - Explain why this research is needed now
   - Describe preliminary observations or pilot data (if available)
   - Build a compelling case for the proposed work

4. Lab Expertise Connection
   - Reference relevant lab capabilities, publications, or expertise
   - Show how the team is uniquely positioned to address these gaps
   - Cite specific resources, methods, or prior work from the lab

WRITING GUIDELINES:
- Use formal academic tone with clear, scholarly language
- Support all claims with specific citations from the provided context
- Build a logical narrative that flows from broad context to specific gaps
- Be precise and factual - avoid speculation or overclaiming
- Aim for 300-500 words with well-structured paragraphs
- Use present tense for established facts, past tense for studies

Use the provided context to support your statements with specific citations.""",

        "Specific Aims": """You are an expert grant writer preparing the SPECIFIC AIMS section for a research proposal.

Grant Topic: {overview}

Write compelling Specific Aims that clearly articulate what you will accomplish. Your writing should:

STRUCTURE REQUIREMENTS:
Generate 2-3 numbered Specific Aims that follow this structure:

AIM [NUMBER]: [Concise title describing the aim]

[Opening sentence stating the aim's central hypothesis or objective]

RATIONALE: [1-2 sentences explaining why this aim is important and how it addresses a gap from the Background]

APPROACH: [2-3 sentences describing the experimental strategy, key methods, and general workflow]

EXPECTED OUTCOMES: [1-2 sentences describing anticipated results and deliverables]

SUCCESS CRITERIA: [1 sentence defining measurable indicators of success]

CONTENT REQUIREMENTS:
1. Each aim should be:
   - Scientifically rigorous with clear hypotheses
   - Achievable within the project timeframe and budget
   - Testable with concrete success metrics
   - Progressive - aims should build logically on each other

2. Approach descriptions should:
   - Reference specific methodologies from the lab's toolkit
   - Be detailed enough to show feasibility
   - Demonstrate alignment with lab expertise
   - Include innovative aspects when relevant

3. Integration and coherence:
   - Aims should work together toward the overall objective
   - Show logical progression and interdependence
   - Address different aspects of the central research question
   - Align with lab capabilities mentioned in the Background

WRITING GUIDELINES:
- Use confident, assertive language (we will determine, we will establish)
- Be specific and concrete - avoid vague generalities
- Include technical details that demonstrate expertise
- Reference lab resources and prior work when applicable
- Aim for 200-400 words total across all aims
- Use active voice and present/future tense

Make aims achievable, testable, and aligned with the lab's demonstrated expertise.""",

        "Significance": """You are an expert grant writer preparing the SIGNIFICANCE section for a research proposal.

Grant Topic: {overview}

Write a compelling Significance section that demonstrates why this research matters. Your writing should:

CONTENT REQUIREMENTS:
1. Scientific Impact and Advancement
   - Explain how this work will advance the field's understanding
   - Describe the scientific knowledge that will be gained
   - Identify which fundamental questions will be answered
   - Position the work within broader scientific goals
   - Explain how results will enable future research directions

2. Clinical and Practical Applications
   - Describe potential translational applications (if applicable)
   - Explain benefits to patients, clinicians, or practitioners
   - Identify specific health problems or challenges that could be addressed
   - Discuss timeline for potential real-world impact
   - Connect to public health priorities when relevant

3. Innovation and Novelty
   - Highlight what makes this approach unique or groundbreaking
   - Contrast with existing methods or paradigms
   - Explain conceptual or technical innovations
   - Describe transformative potential of the work
   - Identify paradigm shifts or new directions this enables

4. Broader Impact
   - Discuss societal benefits beyond immediate scientific community
   - Address economic, policy, or educational implications
   - Consider interdisciplinary impact and applications
   - Explain long-term vision and future directions
   - Connect to national priorities or strategic initiatives

5. Timeliness and Urgency
   - Explain why this research is needed now
   - Describe recent developments that make this work timely
   - Identify windows of opportunity or emerging challenges
   - Position work relative to field momentum

WRITING GUIDELINES:
- Be aspirational yet realistic about potential impact
- Support claims with evidence from the provided context
- Use compelling language that conveys importance and urgency
- Balance immediate and long-term significance
- Address multiple dimensions of impact (scientific, clinical, societal)
- Aim for 250-400 words with clear paragraph structure
- Avoid overclaiming or unsupported speculation

Support all significance claims with evidence from the provided context.""",

        "Innovation": """You are an expert grant writer preparing the INNOVATION section for a research proposal.

Grant Topic: {overview}

Write a compelling Innovation section that highlights novel aspects of this research. Your writing should:

CONTENT REQUIREMENTS:
1. Novel Concepts and Approaches
   - Identify conceptual innovations and new theoretical frameworks
   - Explain paradigm shifts or new ways of thinking about the problem
   - Describe unique hypotheses or models being tested
   - Highlight creative connections between disparate ideas
   - Explain how this challenges current assumptions

2. Methodological Innovation
   - Describe new techniques, tools, or technologies to be developed
   - Explain novel combinations of existing methods
   - Highlight technical advances over current state-of-the-art
   - Detail innovative experimental designs or analytical approaches
   - Describe unique resources or capabilities being leveraged

3. Contrast with Existing Work
   - Explicitly compare to current standard approaches
   - Identify limitations of existing methods that are being overcome
   - Explain why previous attempts have been insufficient
   - Use citations to establish what is state-of-the-art
   - Clearly articulate the innovative leap forward

4. Transformative Potential
   - Describe how success will change the field
   - Identify new research directions this will enable
   - Explain broader applicability beyond immediate focus
   - Discuss potential for unexpected discoveries
   - Consider long-term transformative impact

5. Risk and Reward
   - Acknowledge innovative aspects that carry higher risk
   - Explain why the potential payoff justifies the risk
   - Describe how the team is positioned to overcome challenges
   - Balance innovation with feasibility

WRITING GUIDELINES:
- Be specific and concrete about what is novel
- Clearly distinguish your approach from existing work
- Use comparative language (unlike, in contrast to, beyond)
- Support claims with citations showing state-of-the-art
- Convey excitement about innovative aspects
- Aim for 200-350 words with focused paragraphs
- Balance bold claims with realistic assessment

Use citations from the provided context to contrast with current state-of-the-art.""",

        "Approach": """You are an expert grant writer preparing the APPROACH section for a research proposal.

Grant Topic: {overview}

Write a detailed Approach section that demonstrates how you will accomplish the Specific Aims. Your writing should:

STRUCTURE REQUIREMENTS:
Organize by Specific Aim with subsections:

AIM [NUMBER]: [Brief title]

RATIONALE: [Why this approach is optimal for addressing this aim]

EXPERIMENTAL DESIGN:
- [Overall design and strategy]
- [Key variables and controls]
- [Sample sizes and replication plans]
- [Expected timelines]

METHODS AND PROCEDURES:
- [Detailed protocols for each major experiment]
- [Specific techniques, instruments, and reagents]
- [Data collection procedures]
- [Reference to established protocols from lab]

DATA ANALYSIS:
- [Statistical methods and justification]
- [Analysis software and tools]
- [Interpretation framework]
- [Success criteria and benchmarks]

POTENTIAL PROBLEMS AND ALTERNATIVES:
- [Anticipated challenges]
- [Alternative strategies if first approach fails]
- [How results will be interpreted if unexpected]

CONTENT REQUIREMENTS:
1. Scientific Rigor
   - Justify all methodological choices
   - Explain controls and how they validate results
   - Describe reproducibility measures
   - Address statistical power and sample size rationale
   - Include data quality assurance plans

2. Technical Detail
   - Provide sufficient detail for expert reviewers to assess feasibility
   - Reference specific protocols, instruments, and resources
   - Cite established methods from the literature
   - Describe any method development or optimization needed
   - Include technical specifications when relevant

3. Lab Expertise Integration
   - Reference lab's prior experience with methods
   - Cite relevant publications or preliminary data
   - Mention collaborations or core facilities available
   - Demonstrate team's qualifications for proposed work

4. Timeline and Feasibility
   - Provide realistic timeline for each aim
   - Show logical progression and dependencies
   - Identify milestones and decision points
   - Demonstrate work is achievable in project period

5. Risk Management
   - Honestly assess potential challenges
   - Provide concrete alternative strategies
   - Show depth of planning and problem-solving
   - Build confidence in successful completion

WRITING GUIDELINES:
- Use clear, organized structure with subsections
- Write in future tense (we will measure, we will analyze)
- Be comprehensive yet concise - every sentence adds value
- Include technical details that demonstrate expertise
- Cite specific protocols or methods from context when applicable
- Aim for 400-600 words with clear sectional organization
- Balance detail with readability

Reference specific protocols, methods, and resources from the provided context.""",

        "Environment": """You are an expert grant writer preparing the ENVIRONMENT section for a research proposal.

Grant Topic: {overview}

Write a compelling Environment section that demonstrates institutional support and resources. Your writing should:

CONTENT REQUIREMENTS:
1. Institutional Resources
   - Describe research facilities and laboratory space
   - Detail major equipment and instrumentation available
   - Highlight specialized core facilities and services
   - Mention computational resources and infrastructure
   - Describe animal facilities, biosafety levels, or specialized spaces

2. Support Personnel and Expertise
   - Identify technical support staff and their qualifications
   - Describe administrative and grants management support
   - Mention statistical or bioinformatics consultation services
   - Highlight research coordinators or project managers
   - Note training resources for students and postdocs

3. Collaborative Network
   - Describe relevant collaborations and partnerships
   - Identify consultants and their expertise areas
   - Mention institutional centers or initiatives
   - Highlight interdisciplinary programs or networks
   - Describe access to patient populations or data (if relevant)

4. Institutional Commitment
   - Demonstrate institutional support for this research
   - Mention relevant strategic initiatives or priorities
   - Describe protected time or cost-sharing (if applicable)
   - Highlight recent institutional investments in relevant areas
   - Note recognition or distinction in relevant field

5. Track Record
   - Reference successful projects from the lab
   - Mention funding history and success rate
   - Cite relevant publications and their impact
   - Demonstrate team's productivity and expertise
   - Show institutional research strengths

WRITING GUIDELINES:
- Be specific about resources - don't just list, explain relevance
- Demonstrate how environment enables the proposed work
- Show integration between different resources and support systems
- Convey institutional commitment and support
- Reference specific facilities and capabilities from context
- Aim for 150-300 words organized by resource type
- Balance breadth of resources with focused relevance to project

Use citations from the provided context to demonstrate lab capabilities and resources.""",

        "Bibliography": """Generate a comprehensive Works Cited section in MLA 9th edition format.

Include all sources referenced throughout the grant proposal sections.

FORMATTING REQUIREMENTS:
1. MLA 9th Edition Style
   - Author(s). "Title of Source." Container, Publication Date.
   - Use hanging indent (first line flush left, subsequent lines indented)
   - Alphabetize by first author's last name
   - Use et al. for 3+ authors

2. Organization
   - Sort all citations alphabetically by author or title
   - Group related citations together when appropriate
   - Ensure every in-text citation has a corresponding entry
   - Include full bibliographic information for each source

3. Source Types
   - Journal articles: Author. "Article Title." Journal Name, vol., no., year, pages.
   - Books: Author. Book Title. Publisher, year.
   - Online sources: Author. "Page Title." Website Name, Publication Date, URL.
   - Reports/Documents: Organization. Report Title. Year.

Generate the complete bibliography based on all citations used in the grant sections."""
    }

    def __init__(self, corpus_manager: GrantCorpusManager, db: GrantDatabase):
        """
        Initialize service with managers.

        Args:
            corpus_manager: GrantCorpusManager instance
            db: GrantDatabase instance
        """
        self.corpus_manager = corpus_manager
        self.db = db
        self.inference = GrantInferenceAdapter()

        logger.info("EnhancedGrantService initialized")

    def generate_section(
        self,
        project_id: str,
        section_name: str,
        overview: str,
        additional_context: str = ""
    ) -> SectionResult:
        """
        Generate a grant section using RAGPipeline + inference.py.

        Process:
        1. Build section-specific query from template
        2. Retrieve context with citations using RAGPipeline
        3. Generate using inference.py with consistent parameters
        4. Assess quality (including citation checks)
        5. Save draft to database with conversation history
        6. Log telemetry

        Args:
            project_id: Project identifier
            section_name: Section name (e.g., "Background")
            overview: Grant overview/description
            additional_context: Optional additional requirements

        Returns:
            SectionResult with generated content and metadata
        """
        logger.info(f"Generating {section_name} for project {project_id}")

        try:
            # Build section-specific query
            query_template = self.SECTION_PROMPTS.get(
                section_name,
                "Generate {section_name} section for a grant on {overview}."
            )

            section_query = query_template.format(
                section_name=section_name,
                overview=overview
            )

            if additional_context:
                section_query += f"\n\nAdditional requirements: {additional_context}"

            # Get project RAG index
            project_index_dir = self.db.get_project_field(project_id, 'index_dir')
            if not project_index_dir:
                raise ValueError(f"Project {project_id} not found or has no index directory")

            # Generate using inference adapter
            content, citations, gen_time = self.inference.generate_section(
                section_query=section_query,
                project_index_dir=project_index_dir,
                conversation_history=None  # First turn
            )

            # Assess quality
            quality_score, feedback = self.assess_section_quality(
                section_name=section_name,
                content=content,
                citations=citations
            )

            # Build conversation history for future refinements
            from inference import LABGPT_SYSTEM
            conversation_history = [
                {"role": "system", "content": LABGPT_SYSTEM},
                {"role": "user", "content": section_query},
                {"role": "assistant", "content": content}
            ]

            # Save draft to database
            draft_id = self.db.save_draft(
                project_id=project_id,
                section_name=section_name,
                content=content,
                citations=citations,
                quality_score=quality_score,
                quality_feedback=feedback,
                generation_params=self.inference.get_parameters_info(),
                conversation_history=conversation_history
            )

            # Log telemetry
            self.db.log_generation(
                project_id=project_id,
                section_name=section_name,
                query=section_query,
                num_chunks=len(citations),
                cache_hits=0,  # Would need to extract from RAG stats
                cache_misses=0,
                time_sec=gen_time
            )

            logger.info(
                f"Generated {section_name} (draft {draft_id}): "
                f"{len(content)} chars, quality={quality_score:.2f}, "
                f"{len(citations)} citations, {gen_time:.2f}s"
            )

            return SectionResult(
                draft_id=draft_id,
                section_name=section_name,
                content=content,
                citations=citations,
                quality_score=quality_score,
                quality_feedback=feedback,
                version=1,
                generation_time=gen_time
            )

        except Exception as e:
            logger.error(f"Error generating {section_name}: {e}", exc_info=True)
            raise

    def refine_section(
        self,
        project_id: str,
        section_name: str,
        feedback: str
    ) -> SectionResult:
        """
        Refine section based on user feedback (multi-turn conversation).

        Process:
        1. Load latest draft and conversation history
        2. Append feedback to conversation
        3. Optionally retrieve new context if feedback suggests it
        4. Generate refinement using multi-turn inference
        5. Assess quality
        6. Save new draft version with parent lineage
        7. Save feedback to database

        Args:
            project_id: Project identifier
            section_name: Section name
            feedback: User's feedback/refinement request

        Returns:
            SectionResult with refined content and metadata
        """
        logger.info(f"Refining {section_name} for project {project_id}")

        try:
            # Get latest draft
            latest_draft = self.db.get_latest_draft(project_id, section_name)
            if not latest_draft:
                raise ValueError(
                    f"No draft found for {section_name} in project {project_id}. "
                    "Generate first before refining."
                )

            # Get conversation history
            conversation_history = latest_draft['conversation_history']

            # Get project index
            project_index_dir = self.db.get_project_field(project_id, 'index_dir')

            # Refine using multi-turn conversation
            refined_content, new_citations, gen_time = self.inference.refine_section(
                feedback=feedback,
                conversation_history=conversation_history,
                project_index_dir=project_index_dir,
                retrieve_new_context=True  # Allow new context retrieval
            )

            # Use new citations if retrieved, otherwise keep old ones
            final_citations = new_citations if new_citations else latest_draft['citations']

            # Assess quality
            quality_score, quality_feedback = self.assess_section_quality(
                section_name=section_name,
                content=refined_content,
                citations=final_citations
            )

            # Update conversation history
            updated_conversation = conversation_history + [
                {"role": "user", "content": f"Feedback: {feedback}"},
                {"role": "assistant", "content": refined_content}
            ]

            # Save new draft version (with parent_draft_id for lineage)
            new_draft_id = self.db.save_draft(
                project_id=project_id,
                section_name=section_name,
                content=refined_content,
                citations=final_citations,
                quality_score=quality_score,
                quality_feedback=quality_feedback,
                generation_params=self.inference.get_parameters_info(),
                parent_draft_id=latest_draft['draft_id'],
                conversation_history=updated_conversation
            )

            # Save feedback
            self.db.add_feedback(latest_draft['draft_id'], feedback)

            logger.info(
                f"Refined {section_name} (draft {new_draft_id}, v{latest_draft['version']+1}): "
                f"quality={quality_score:.2f}, {gen_time:.2f}s"
            )

            return SectionResult(
                draft_id=new_draft_id,
                section_name=section_name,
                content=refined_content,
                citations=final_citations,
                quality_score=quality_score,
                quality_feedback=quality_feedback,
                version=latest_draft['version'] + 1,
                generation_time=gen_time
            )

        except Exception as e:
            logger.error(f"Error refining {section_name}: {e}", exc_info=True)
            raise

    def assess_section_quality(
        self,
        section_name: str,
        content: str,
        citations: List[Dict]
    ) -> Tuple[float, str]:
        """
        Assess section quality with citation checks.

        Scoring (0.0-1.0):
        - Length appropriateness
        - Section-specific requirements
        - Citation adequacy
        - Content structure

        Args:
            section_name: Section name
            content: Generated content
            citations: List of citation dicts

        Returns:
            Tuple of (quality_score, feedback_text)
        """
        feedback_parts = []
        score = 1.0  # Start with perfect score

        # Length check
        word_count = len(content.split())

        min_words = {
            "Background": 300,
            "Specific Aims": 200,
            "Significance": 250,
            "Innovation": 200,
            "Approach": 400,
            "Environment": 150,
            "Bibliography": 50
        }.get(section_name, 200)

        if word_count < min_words:
            score -= 0.2
            feedback_parts.append(
                f"Section is short ({word_count} words). "
                f"Aim for at least {min_words} words for {section_name}."
            )
        elif word_count > 1000:
            score -= 0.1
            feedback_parts.append(
                f"Section is long ({word_count} words). "
                "Consider being more concise."
            )

        # Citation check (NEW: adapted for modern system)
        if section_name != "Bibliography":
            if len(citations) < 2:
                score -= 0.15
                feedback_parts.append(
                    "Add more citations to support claims (minimum 2 sources)."
                )

            if word_count > 500 and len(citations) < 5:
                score -= 0.1
                feedback_parts.append(
                    "Longer sections should reference more diverse sources (5+ recommended)."
                )

        # Section-specific checks
        content_lower = content.lower()

        if section_name == "Background":
            if "gap" not in content_lower and "limitation" not in content_lower:
                score -= 0.1
                feedback_parts.append(
                    "Background should identify research gaps or limitations."
                )

        elif section_name == "Specific Aims":
            if not ("aim 1" in content_lower or "aim one" in content_lower):
                score -= 0.2
                feedback_parts.append(
                    "Specific Aims should be clearly numbered (Aim 1, Aim 2, etc.)."
                )

            # Check for multiple aims
            aim_count = content_lower.count("aim ")
            if aim_count < 2:
                score -= 0.1
                feedback_parts.append(
                    "Consider including 2-3 specific aims."
                )

        elif section_name == "Significance":
            if "impact" not in content_lower:
                score -= 0.1
                feedback_parts.append(
                    "Significance should discuss the impact of the work."
                )

        elif section_name == "Innovation":
            if "novel" not in content_lower and "innovative" not in content_lower:
                score -= 0.1
                feedback_parts.append(
                    "Innovation section should highlight novel aspects."
                )

        elif section_name == "Approach":
            if "method" not in content_lower and "protocol" not in content_lower:
                score -= 0.1
                feedback_parts.append(
                    "Approach should describe specific methods or protocols."
                )

        # Ensure score is in valid range
        score = max(0.0, min(1.0, score))

        # Overall quality assessment
        if score >= 0.9:
            feedback_parts.insert(0, "Excellent quality - ready for review!")
        elif score >= 0.75:
            feedback_parts.insert(0, "Good quality with minor improvements suggested.")
        elif score >= 0.6:
            feedback_parts.insert(0, "Acceptable quality - consider addressing suggestions.")
        else:
            feedback_parts.insert(0, "Significant improvements recommended.")

        return score, "\n".join(feedback_parts)

    def get_project_progress(self, project_id: str) -> Dict:
        """
        Get completion progress for all sections in a project.

        Args:
            project_id: Project identifier

        Returns:
            Dict with section_name -> status for each section
        """
        sections_status = {}

        for section_name in self.SECTION_PROMPTS.keys():
            latest_draft = self.db.get_latest_draft(project_id, section_name)

            if latest_draft:
                sections_status[section_name] = {
                    'completed': True,
                    'quality_score': latest_draft['quality_score'],
                    'version': latest_draft['version'],
                    'word_count': len(latest_draft['content'].split()),
                    'citation_count': len(latest_draft['citations'])
                }
            else:
                sections_status[section_name] = {
                    'completed': False,
                    'quality_score': 0.0,
                    'version': 0,
                    'word_count': 0,
                    'citation_count': 0
                }

        return sections_status
