import os
import io
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, session, send_file, flash
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import rag_service  # Make sure rag_service.py is in the same folder

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Change to a secure random key in production

# Allowed upload extensions
ALLOWED_EXTENSIONS = {'txt'}

# Define the sections order for the grant proposal
SECTIONS_ORDER = [
    "Background", 
    "Objectives", 
    "Specific Aims", 
    "Methods", 
    "Preliminary work", 
    "Impact/Relevance",
    "Feasibility, Risks and Mitigation Strategies",
    "Project Outcomes and Future Directions",
    "Research Data Management and Open Science",
    "Expertise, Experience, and Resources",
    "Summary of Progress of Principal Investigator",
    "Lay Abstract",
    "Lay Summary"
]

# Define personalized prompts for each section
SECTION_PROMPTS = {
    "Background": """Using the provided project overview and relevant information from supporting documents, write a comprehensive Background section for the grant proposal. This section should establish the context and justification for the research. Focus on:

    Broad Context: Summarize the current state of knowledge in the field, citing key findings or established facts from the literature (as given in the documents).

    Gap/Problem Statement: Identify the specific gap in knowledge or problem that the proposed research will address, explaining why this gap is important.

    Significance: Persuade the reader of the importance of addressing this problem, highlighting any pressing needs or opportunities (e.g. scientific, societal, or practical relevance).

Ensure the narrative flows logically from general background to the specific problem, and use a formal, scholarly tone. The Background should clearly set the stage for the Objectives that follow.""",

    "Objectives": """Drawing on the grant overview and any pertinent details from the Background, articulate the Objectives of the proposed project in a clear and compelling manner. The Objectives section should:

    Overall Goals: State the broad, high-level goals of the research, reflecting the ultimate aims of the project. (For example, what do you hope to achieve or discover?)

    Specific Objectives: Break down the general goal into 2-3 concrete objectives or questions the project will address. These should directly tackle the problem identified in the Background.

    Alignment and Importance: Explain how achieving these objectives will contribute to the field or solve the stated problem, reinforcing the project's importance.

Write this in an expository, persuasive style. Each objective should be phrased as an outcome (what will be accomplished) and, if appropriate, be SMART (Specific, Measurable, Attainable, Realistic, Time-bound) in the context of the project.""",

    "Specific Aims": """Using the defined Objectives and supporting document insights, develop the Specific Aims section of the grant proposal. This section should distill the project's plan into a set of tangible aims or research questions. Ensure that:

    Aim Structure: You present 2–4 numbered aims. Each aim is a concise statement of a specific goal or hypothesis that will be tested or achieved. (For example: Aim 1: Investigate the effect of X on Y…)

    Detail and Scope: For each aim, include a brief description of what will be done to accomplish it – for instance, the approach or experiment associated with that aim – without delving fully into Methods. Emphasize how each aim addresses part of the overall objectives.

    Outcome: Optionally, mention the expected outcome or what success looks like for each aim (if this helps clarify the aim's role).

Keep the tone formal and focused. The Specific Aims should logically flow from the Objectives, and they should be realistic and achievable within the project's scope. If available, incorporate any relevant preliminary insights from documents that justify an aim (e.g. an observation or finding that led to that aim).""",

    "Methods": """Based on the project's aims and retrieved methodological details from the documents, write the Methods section of the proposal. This section should describe how the research will be carried out. Include the following:

    Overview of Approach: Begin with a summary of the overall research design or strategy (e.g. experimental, observational, theoretical, or computational approach).

    Detailed Methodology: For each Specific Aim (or project component), describe the methods, techniques, and procedures you will use. Be sure to mention:

        Data Collection: What data will be collected and how (e.g. instruments, surveys, experiments, lab protocols)?

        Analysis: How the data will be analyzed to address the research questions (e.g. statistical tests, computational models, qualitative analysis strategies).

        Tools/Techniques: Any specific tools, technologies, or protocols (from the supporting docs or standard practice) that will be employed, with justification for their use.

    Timeline/Workplan: (If appropriate) note the sequence or timeline of the work, possibly referencing phases for each aim.

The writing should be detailed and precise, suitable for expert evaluation. Incorporate evidence from supporting documents to strengthen the methodology description (for example, reference a preliminary experiment setup from a document or established methods from prior studies). Cite any established methodology if it's mentioned in the provided context. Ensure the tone remains persuasive by highlighting why these methods are the most effective means to achieve the aims.""",

    "Preliminary work": """Using the lab's previous findings and insights from the documents, compose the Preliminary Work and Impact/Relevance section. This section has two roles: demonstrating what has already been done and arguing why the project matters. Make sure to:

    Preliminary Results/Work: Summarize any preliminary studies, pilot experiments, or prior work by the research team that relate to this project. Describe these results or experiences, indicating how they support the feasibility of the proposed research. (For instance, "Our lab has shown in a pilot study that X…," or "Initial data from [supporting document] indicate Y.") Use quantitative results if available (e.g. key data points from prior experiments) to add credibility.

    Impact and Relevance: Explain the potential impact of the proposed research. Discuss how successful completion of the project will advance scientific knowledge, address a critical need, or benefit society. Tie this to the earlier Background problem and Objectives, reinforcing why the work is worth doing. If applicable, also note alignment with the funding agency's priorities or broader initiatives (using hints from the overview/documents).

    Connecting Preliminary to Proposal: Emphasize how the preliminary work just described has laid the groundwork for the new proposal and increases the likelihood of success.

Maintain an enthusiastic yet scholarly tone. Use the supporting documents to pull any evidence of prior success or existing capacity (e.g. published papers, previous project outcomes) to bolster the argument. The goal is to convince the reviewer that the team is building on solid foundations and that the project will have meaningful outcomes.""",

    "Impact/Relevance": """Using the lab's previous findings and insights from the documents, compose the Preliminary Work and Impact/Relevance section. This section has two roles: demonstrating what has already been done and arguing why the project matters. Make sure to:

    Preliminary Results/Work: Summarize any preliminary studies, pilot experiments, or prior work by the research team that relate to this project. Describe these results or experiences, indicating how they support the feasibility of the proposed research. (For instance, "Our lab has shown in a pilot study that X…," or "Initial data from [supporting document] indicate Y.") Use quantitative results if available (e.g. key data points from prior experiments) to add credibility.

    Impact and Relevance: Explain the potential impact of the proposed research. Discuss how successful completion of the project will advance scientific knowledge, address a critical need, or benefit society. Tie this to the earlier Background problem and Objectives, reinforcing why the work is worth doing. If applicable, also note alignment with the funding agency's priorities or broader initiatives (using hints from the overview/documents).

    Connecting Preliminary to Proposal: Emphasize how the preliminary work just described has laid the groundwork for the new proposal and increases the likelihood of success.

Maintain an enthusiastic yet scholarly tone. Use the supporting documents to pull any evidence of prior success or existing capacity (e.g. published papers, previous project outcomes) to bolster the argument. The goal is to convince the reviewer that the team is building on solid foundations and that the project will have meaningful outcomes.""",

    "Feasibility, Risks and Mitigation Strategies": """Write the Feasibility, Risks, and Mitigation Strategies section, drawing on the project plan and any risk assessments or supporting evidence from the documents. In this section:

    Feasibility: First, argue that the project is feasible. Highlight the strengths that enable success: the team's expertise (mention relevant skills or past accomplishments), the adequacy of the preliminary data, availability of necessary resources/technology, and any supportive data from documents that show the approach can work.

    Potential Risks/Challenges: Identify the main risks or challenges that could arise in the project. These might include technical difficulties, methodological limitations, resource constraints, or possible experimental outcomes that could complicate interpretation. For each risk, be specific (e.g., "a possible risk is that the sample size may be insufficient to detect X…" or "the Y assay may have low sensitivity for Z").

    Mitigation Strategies: For every risk mentioned, describe a plan to mitigate or address it. This could involve alternative approaches (a Plan B), additional experiments, protocol adjustments, or expert collaborations/consultations. For example, "If X approach fails to yield sufficient data, we will use an alternative method Y as described in [document reference]" or "To mitigate risk Z, we will perform a preliminary test at smaller scale before full deployment."

    Confidence Statement: Conclude by reinforcing that, given these strategies, the project remains highly likely to succeed, and any foreseeable hurdles are manageable.

Keep the tone confident and proactive, showing reviewers that the team has thoughtfully prepared for challenges. Incorporate any relevant info from the documents (like past troubleshooting successes, or contingency plans used in related projects) to make the section concrete and credible.""",

    "Project Outcomes and Future Directions": """Using the proposal details and context from the overview/documents, write the Project Outcomes and Future Directions section. This section should discuss what will happen after or as a result of the project. Include:

    Expected Outcomes: Describe the key results or products expected from the project. This may include new data, theoretical insights, prototypes, publications, or any deliverables. Be specific: for example, "We expect to identify __," or "The project will result in a dataset/methodology…". Make sure these outcomes tie back to the Objectives and Aims, demonstrating that each goal will yield something concrete.

    Impact of Outcomes: Explain how these outcomes will impact the field or broader context. Will they open new research questions, provide evidence for policy, enable a new technology, etc.? Highlight the value of the outcomes using evidence or projections (for instance, references from literature about what such an outcome could enable).

    Future Directions: Outline how this work will pave the way for future research or applications. This could mean follow-up studies to explore new questions that arise, steps toward clinical trials or commercialization (if applicable), or how the findings lay a foundation for the PI's long-term research agenda. Mention any specific future projects or grant applications that could logically follow.

    Sustainability/Next Steps: If relevant, note how the results will be sustained or taken forward beyond the grant period (e.g. data will be made available for other researchers, collaborations will continue, or an outlined plan for scaling up the research).

Maintain an optimistic and visionary tone. This section should leave reviewers with a sense that funding this project will have lasting benefits and momentum. Use information from the grant overview and documents to ground future plans in reality (e.g., if a document indicates a long-term research goal of the lab, align with that).""",

    "Research Data Management and Open Science": """Craft the Research Data Management and Open Science section, describing how project data and outputs will be handled, stored, and shared, in line with best practices and any guidelines found in the documents. Ensure the section covers:

    Data Management Plan: Outline how data will be collected, stored securely, and organized. Specify the formats of data, approximate volume (if known), and the tools or infrastructure for storage (e.g., institutional servers, cloud storage). Mention data backup routines and measures to ensure data integrity (e.g., regular backups, version control).

    Ethics and Privacy: If the project involves sensitive data (e.g., human subjects, personal data), state how you will maintain confidentiality and comply with regulations. This might include anonymization procedures, consent management, or ethics board approvals (reference any standards from documents or institutional policies).

    Data Sharing and Open Science: Describe how you will share the data and results with the broader community in an open-science manner. For example, indicate which data (or perhaps all data) will be made publicly available, and through what platform or repository (e.g., an open data repository, supplementary info in publications). Mention sharing timelines (e.g., after publication or after project completion) and any embargo if necessary. Also discuss sharing other outputs: will you publish in open-access journals or preprint servers? Will you release code or software under an open-source license?

    Standards and FAIR Principles: (If relevant) note that you will adhere to community standards for metadata and data formatting, ensuring data are Findable, Accessible, Interoperable, and Reusable (FAIR). For instance, mention using standardized file formats and metadata schemas as recommended in your field (cite any guidelines from the documents or funder).

Write this section in a confident, factual tone. The goal is to assure reviewers that the project's data and knowledge will be managed responsibly and shared to maximize impact. Use any document-provided specifics (like institutional policies or previous data management examples) to add detail. For example, "Data will be deposited in the XYZ database, as our lab has done for previous projects.""",

    "Expertise, Experience, and Resources": """Using information from the grant overview and uploaded documents (e.g., CVs, facility descriptions, past project descriptions), compose the Expertise, Experience, and Resources section. This part should demonstrate that the team and institution have the necessary background and tools to complete the project successfully. Be sure to cover:

    Team Expertise: Introduce the principal investigator (PI) and key team members, summarizing their relevant experience and qualifications. Highlight specific expertise that aligns with the project's topic (for example, "Dr. X has 10 years of experience in Y and has published Z papers on related topics"). If documents include prior publications or achievements, reference them to bolster credibility.

    Track Record: Mention any notable accomplishments of the team related to the proposal's field – such as previous grants, publications, patents, or innovations. Emphasize successes that show an ability to deliver results (e.g., "Previously developed a prototype system…" or "discovered…," from supporting docs).

    Institutional Resources: Describe the facilities, equipment, and resources available. This can include laboratory space, specialized equipment, computing resources, access to core facilities, libraries, databases, or any unique resource that gives an advantage. For example, "The lab is equipped with a state-of-the-art ___, as documented in [facility document], enabling the proposed experiments."

    Collaborations/Support: If applicable, note any collaborations or consultants who will contribute specific expertise, and any institutional support (like funding, mentoring, technical staff, or matching funds provided outside the grant). Use the documents to find any letters of support or partnership details to mention briefly.

    Alignment: Conclude by reinforcing how the combined expertise and resources form a strong foundation for the project.

The tone should be confident and factual. Essentially, this section should convince reviewers that this team is exceptionally well-prepared to execute the project. Leverage the content of any provided CVs, biosketches, or institutional descriptions from the uploaded materials to add concrete details (names, years of experience, specific tools or labs, etc.).""",

    "Summary of Progress of Principal Investigator": """Compile the Summary of Progress of the Principal Investigator (PI) section, using the PI's past project reports and achievements (from the documents or overview). This section should summarize the PI's recent research progress, especially if this proposal is a continuation or related to prior funded work. Include:

    Previous Project(s) Overview: Briefly recap the PI's most recent funded project(s) or research endeavors. Focus on those relevant to the current proposal. For example, mention the title or topic of the previous grant and its goals.

    Accomplishments and Results: Summarize what the PI achieved in those projects. Highlight outcomes such as key findings, publications, patents, developed techniques, or any objectives that were met. If quantitative outcomes are available (e.g., "published 3 papers, trained 2 students, developed a prototype"), include them. Use a narrative form or bullet list for clarity.

    Impact of Previous Work: Describe how the PI's prior work has contributed to the field or influenced the current proposal. For instance, "The findings from the previous project (as detailed in the attached progress report) led to new questions addressed by this proposal," or "The PI's earlier work on X established a methodology that will be applied here." Show that the PI builds on past success.

    Growth and Development: Optionally, note any growth in the PI's expertise or the lab's capacity as a result of previous projects (e.g., new skills learned, new collaborations formed, new techniques mastered). This can demonstrate an increasing capability.

    Conclusion/Relation to New Proposal: Conclude by explicitly linking the past progress to the current application's goals, stating that the PI's proven track record of delivering results bodes well for the success of the proposed project.

Maintain a tone of factual confidence and pride (without being boastful). The section should read like a mini progress report – concise and focused on evidence of productivity. Use specifics from the PI's past progress documents (such as year-by-year achievements or quotes from evaluations, if available) to ensure accuracy. This section essentially assures reviewers that the PI has successfully managed past research efforts and will do so again.""",

    "Lay Abstract": """Using non-technical language, write a Lay Abstract for the grant proposal that could be understood by a general audience with no special knowledge of the field. This abstract should be brief (around one paragraph, typically 3-5 sentences) and cover the core elements of the project:

    What the project is about (Big Picture): Introduce the general area or problem in a way that anyone can appreciate. (E.g., "Cancer affects millions of people… This project explores a new approach to improve treatment.")

    What the project will do (Objective/Approach): Explain in simple terms what the researchers are planning to do. Focus on the main objective or question and the approach, avoiding jargon. (For example, "We will study how X works by doing Y," phrased in lay terms.)

    Why it matters (Impact): State the potential benefit or significance of the research for society or the field, in a relatable way. (E.g., "This could lead to better vaccines in the future," or "Understanding X might help scientists develop new energy sources.")

Ensure the tone is engaging, clear, and concise. Do not include technical details or acronyms; if a technical term is unavoidable, briefly define it in simple words. The Lay Abstract should excite interest and understanding, giving a snapshot of the project's essence and importance to someone unfamiliar with the subject. It often helps to imagine explaining the project to a friend or family member who is not a scientist. Keep it under ~300 words (or as required), focusing on the highlights of the proposal.""",

    "Lay Summary": """Prepare a Lay Summary of the grant proposal for a general audience, expanding on the lay abstract in a bit more detail (e.g. one to two short paragraphs). The Lay Summary should be written in clear, accessible language and should cover:

    Introduction to the Problem: Start by describing the issue or question the project addresses in everyday terms, and why it's important. You might include a relatable example or a statistic (if it can be stated simply) from the documents to illustrate the problem.

    Project Description: Explain what the project will do to address this problem. Break it down step-by-step in plain language. For instance, describe the approach without technical jargon: "The researchers will compare X and Y," or "They will create a new system to …," etc. Ensure this is a slightly fuller explanation than the lay abstract, but still understandable by non-experts.

    Expected Outcome and Benefit: Describe what outcomes the project hopes to achieve and why that's good. Emphasize how the results might affect real-world situations or advance knowledge in a way that the public can appreciate. For example, "If successful, this research could lead to… (a cure, a new technology, better understanding of…)."

    Closing Statement: End with a hopeful or broad statement about the significance of the work. (E.g., "In summary, this project aims to …, which could ultimately help ….")

Keep the tone friendly and informative, as if writing a short article for a news outlet or a summary for a funding agency's public brief. Avoid technical terms; when unavoidable, explain them in simple words. Use the grant overview and technical sections as a basis, but translate all jargon into layperson terms. The Lay Summary should stand on its own as a clear explanation of the project to someone who has no background in the subject. Aim for coherence and enthusiasm, conveying the excitement of the research and its value to society."""
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------------------
# Helper: Retrieve relevant context from uploaded documents
# -------------------------------
def retrieve_uploaded_context(query, docs):
    """
    Given a query and a list of document texts (from uploaded files), split them into paragraphs,
    compute embeddings (using the same SentenceTransformer from rag_service), and return the top 3 paragraphs.
    """
    if not docs:
        return ""
    paragraphs = []
    for doc in docs:
        # Split by double newline (assumes paragraphs are separated by blank lines)
        paras = doc.split("\n\n")
        paragraphs.extend([p.strip() for p in paras if p.strip()])
    if not paragraphs:
        return ""
    # Compute embeddings
    query_emb = rag_service.embed_model.encode([query])
    paras_emb = rag_service.embed_model.encode(paragraphs)
    # Normalize embeddings
    query_norm = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    paras_norm = paras_emb / np.linalg.norm(paras_emb, axis=1, keepdims=True)
    similarities = np.dot(paras_norm, query_norm.T).squeeze()
    # Get indices of top 3 similar paragraphs
    top_indices = np.argsort(similarities)[-3:][::-1]
    top_paras = [paragraphs[i] for i in top_indices]
    return "\n".join(top_paras)

# -------------------------------
# Helper: Generate section text using RAG service
# -------------------------------
def generate_section_text(section_name, mod_request=''):
    overview = session.get('overview', '')
    uploaded_docs = session.get('uploaded_docs', [])
    # Retrieve context from uploaded documents (if any)
    uploaded_context = retrieve_uploaded_context(f"Extract relevant information for the {section_name} section", uploaded_docs)
    # Get previously saved sections (if any) in the correct order
    current_index = SECTIONS_ORDER.index(section_name)
    previous_sections = [session['sections'].get(sec, '') for sec in SECTIONS_ORDER[:current_index]]
    previous_text = "\n".join(previous_sections)
    
    # Build the comprehensive context for the LLM
    context_parts = []
    context_parts.append(f"Grant Overview: {overview}")
    
    if previous_text.strip():
        context_parts.append(f"Previously written sections:\n{previous_text}")
    
    if uploaded_context.strip():
        context_parts.append(f"Supporting documents context: {uploaded_context}")
    
    if mod_request.strip():
        context_parts.append(f"Modification instructions: {mod_request}")
    
    # Get the personalized prompt for this section
    section_prompt = SECTION_PROMPTS.get(section_name, f"Write the '{section_name}' section for a grant proposal in a clear, structured, and professional manner.")
    
    # Combine context and personalized prompt
    full_context = "\n\n".join(context_parts)
    query_prompt = f"{full_context}\n\nTask: {section_prompt}"
    
    # Use the RAG service to generate the text
    generated_text = rag_service.get_rag_answer(query_prompt)
    # Remove any leading boilerplate (like "Answer:") if present
    if "Answer:" in generated_text:
         generated_text = generated_text.split("Answer:")[-1].strip()
    return generated_text

# -------------------------------
# Routes
# -------------------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
         overview = request.form.get('overview')
         files = request.files.getlist('supporting_docs')
         uploaded_texts = []
         for file in files:
             if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                content = file.read().decode('utf-8')
                uploaded_texts.append(content)
         session['overview'] = overview
         session['uploaded_docs'] = uploaded_texts
         session['sections'] = {}  # initialize sections storage
         # Redirect to the first section
         return redirect(url_for('section', section_name='Background'))
    return render_template('index.html')

@app.route('/section/<section_name>', methods=['GET', 'POST'])
def section(section_name):
    if section_name not in SECTIONS_ORDER:
         flash("Invalid section.")
         return redirect(url_for('index'))
    if request.method == 'POST':
         action = request.form.get('action')
         if action == 'Generate':
             mod_request = request.form.get('mod_request', '')
             generated_text = generate_section_text(section_name, mod_request)
             session['sections'][section_name] = generated_text
         elif action == 'Request Changes':
             mod_request = request.form.get('mod_request', '')
             generated_text = generate_section_text(section_name, mod_request)
             session['sections'][section_name] = generated_text
         elif action == 'Save':
             # Save any manual edits made by the user
             current_text = request.form.get('section_text')
             session['sections'][section_name] = current_text
             # Move to the next section if available
             current_index = SECTIONS_ORDER.index(section_name)
             if current_index < len(SECTIONS_ORDER) - 1:
                 next_section = SECTIONS_ORDER[current_index + 1]
                 return redirect(url_for('section', section_name=next_section))
             else:
                 return redirect(url_for('finalize'))
         current_text = session['sections'].get(section_name, '')
         return render_template('section.html', section_name=section_name, section_text=current_text)
    else:
         current_text = session['sections'].get(section_name, '')
         return render_template('section.html', section_name=section_name, section_text=current_text)

@app.route('/finalize', methods=['GET'])
def finalize():
    sections = session.get('sections', {})
    overview = session.get('overview', '')
    return render_template('finalize.html', overview=overview, sections=sections)

@app.route('/download', methods=['GET'])
def download():
    # Generate a PDF of the final grant proposal using ReportLab
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, y, "Grant Proposal")
    y -= 30
    p.setFont("Helvetica", 12)
    p.drawString(50, y, "Overview:")
    y -= 20
    overview = session.get('overview', '')
    for line in overview.splitlines():
         p.drawString(60, y, line)
         y -= 15
         if y < 50:
             p.showPage()
             y = height - 50
    sections = session.get('sections', {})
    for section, text in sections.items():
         y -= 20
         p.setFont("Helvetica-Bold", 14)
         p.drawString(50, y, section + ":")
         y -= 20
         p.setFont("Helvetica", 12)
         for line in text.splitlines():
              p.drawString(60, y, line)
              y -= 15
              if y < 50:
                  p.showPage()
                  y = height - 50
    p.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="grant_proposal.pdf", mimetype='application/pdf')

if __name__ == '__main__':
    app.run(debug=True)
