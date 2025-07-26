#!/usr/bin/env python3
"""
enhanced_app.py

Enhanced Grant Generation Web Application with:
- Improved user interface and experience
- Two-stage generation pipeline
- Quality assessment and feedback
- Progress tracking
- Better document handling
"""

import os
import io
import json
import tempfile
from pathlib import Path
from typing import List, Dict
from flask import Flask, request, render_template, redirect, url_for, session, send_file, flash, jsonify
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import logging

from enhanced_document_processor import EnhancedDocumentProcessor, ProcessedDocument
from enhanced_grant_service import EnhancedGrantService, SectionContent
from session_manager import session_manager

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'doc'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize services
document_processor = EnhancedDocumentProcessor()
grant_service = EnhancedGrantService()

# Grant sections in logical order
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

logging.basicConfig(level=logging.INFO)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_files(files) -> List[str]:
    """Save uploaded files and return file paths"""
    file_paths = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            file_paths.append(filepath)
    return file_paths

def get_session_id():
    """Get or create session ID"""
    if 'session_id' not in session:
        session['session_id'] = session_manager.generate_session_id()
    return session['session_id']

def get_session_data(key: str = None):
    """Get session data from file storage"""
    session_id = get_session_id()
    data = session_manager.load_session_data(session_id) or {}
    return data.get(key) if key else data

def set_session_data(key: str, value):
    """Set session data to file storage"""
    session_id = get_session_id()
    session_manager.update_session_data(session_id, {key: value})

def update_session_data(updates: dict):
    """Update multiple session data fields"""
    session_id = get_session_id()
    session_manager.update_session_data(session_id, updates)

@app.route('/', methods=['GET', 'POST'])
def index():
    """Enhanced home page with document upload and overview"""
    if request.method == 'POST':
        # Get grant overview
        grant_overview = request.form.get('overview', '').strip()
        if not grant_overview:
            flash('Please provide a grant overview.', 'error')
            return render_template('enhanced_index.html')
        
        # Handle file uploads
        uploaded_files = request.files.getlist('supporting_docs')
        file_paths = save_uploaded_files(uploaded_files)
        
        # Process documents
        processed_docs = []
        if file_paths:
            try:
                processed_docs = document_processor.process_multiple_documents(file_paths)
                flash(f'Successfully processed {len(processed_docs)} documents.', 'success')
            except Exception as e:
                flash(f'Error processing documents: {str(e)}', 'error')
                logging.error(f"Document processing error: {e}")
        
        # Store in file-based session (to avoid cookie size limits)
        update_session_data({
            'grant_overview': grant_overview,
            'processed_docs': [
                {
                    'filename': doc.filename,
                    'summary': doc.summary,
                    'document_type': doc.document_type,
                    'word_count': doc.word_count,
                    'sections': doc.sections
                } for doc in processed_docs
            ],
            'current_section_index': 0,
            'completed_sections': {}
        })
        
        # Redirect to first section
        return redirect(url_for('section_workflow', section_name=SECTIONS_ORDER[0]))
    
    return render_template('enhanced_index.html')

@app.route('/section/<section_name>', methods=['GET', 'POST'])
def section_workflow(section_name):
    """Enhanced section editing workflow"""
    grant_overview = get_session_data('grant_overview')
    if not grant_overview:
        flash('Please start by providing a grant overview.', 'error')
        return redirect(url_for('index'))
    
    if section_name not in SECTIONS_ORDER:
        flash('Invalid section name.', 'error')
        return redirect(url_for('index'))
    
    # Get current section index
    current_index = SECTIONS_ORDER.index(section_name)
    set_session_data('current_section_index', current_index)
    
    # Reconstruct processed documents from session
    processed_docs = []
    processed_docs_data = get_session_data('processed_docs') or []
    for doc_data in processed_docs_data:
        processed_docs.append(ProcessedDocument(
            filename=doc_data['filename'],
            text="",  # Not needed for section generation
            sections=doc_data['sections'],
            summary=doc_data['summary'],
            document_type=doc_data['document_type'],
            word_count=doc_data['word_count'],
            is_scanned=False
        ))
    
    # Handle POST request (generation or editing)
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'generate':
            # Generate section content
            try:
                section_content = grant_service.generate_complete_section(
                    processed_docs, section_name, grant_overview
                )
                
                # Store generated content in session
                set_session_data(f'section_{section_name}', {
                    'generated_text': section_content.generated_text,
                    'extracted_info': section_content.extracted_info,
                    'rag_context': section_content.rag_context,
                    'quality_score': section_content.quality_score,
                    'suggestions': section_content.suggestions
                })
                
                flash(f'Generated {section_name} section (Quality Score: {section_content.quality_score:.1f}/5.0)', 'success')
                
            except Exception as e:
                flash(f'Error generating section: {str(e)}', 'error')
                logging.error(f"Section generation error: {e}")
        
        elif action == 'refine':
            # Refine with user feedback
            user_feedback = request.form.get('feedback', '').strip()
            section_data = get_session_data(f'section_{section_name}')
            if user_feedback and section_data:
                try:
                    # Create SectionContent object from session data
                    current_content = SectionContent(
                        section_name=section_name,
                        generated_text=section_data['generated_text'],
                        extracted_info=section_data['extracted_info'],
                        rag_context=section_data['rag_context'],
                        quality_score=section_data['quality_score'],
                        suggestions=section_data['suggestions']
                    )
                    
                    # Refine the content
                    refined_content = grant_service.refine_section_with_feedback(
                        current_content, user_feedback, grant_overview
                    )
                    
                    # Update session
                    set_session_data(f'section_{section_name}', {
                        'generated_text': refined_content.generated_text,
                        'extracted_info': refined_content.extracted_info,
                        'rag_context': refined_content.rag_context,
                        'quality_score': refined_content.quality_score,
                        'suggestions': refined_content.suggestions
                    })
                    
                    flash(f'Refined {section_name} section based on feedback', 'success')
                    
                except Exception as e:
                    flash(f'Error refining section: {str(e)}', 'error')
                    logging.error(f"Section refinement error: {e}")
        
        elif action == 'save':
            # Save current section content
            section_text = request.form.get('section_text', '').strip()
            if section_text:
                completed_sections = get_session_data('completed_sections') or {}
                completed_sections[section_name] = section_text
                set_session_data('completed_sections', completed_sections)
                flash(f'Saved {section_name} section', 'success')
                
                # Move to next section if available
                if current_index < len(SECTIONS_ORDER) - 1:
                    next_section = SECTIONS_ORDER[current_index + 1]
                    return redirect(url_for('section_workflow', section_name=next_section))
                else:
                    # All sections completed, go to finalization
                    return redirect(url_for('finalize'))
    
    # Get current section data
    section_data = get_session_data(f'section_{section_name}') or {}
    completed_sections = get_session_data('completed_sections') or {}
    
    # Calculate progress
    progress = {
        'current_section': section_name,
        'current_index': current_index,
        'total_sections': len(SECTIONS_ORDER),
        'completed_count': len(completed_sections),
        'percentage': (len(completed_sections) / len(SECTIONS_ORDER)) * 100
    }
    
    return render_template('enhanced_section.html', 
                         section_name=section_name,
                         section_data=section_data,
                         progress=progress,
                         sections_order=SECTIONS_ORDER,
                         completed_sections=completed_sections)

@app.route('/navigate/<direction>')
def navigate_section(direction):
    """Navigate between sections"""
    current_index = get_session_data('current_section_index') or 0
    
    if direction == 'next' and current_index < len(SECTIONS_ORDER) - 1:
        next_section = SECTIONS_ORDER[current_index + 1]
        return redirect(url_for('section_workflow', section_name=next_section))
    elif direction == 'prev' and current_index > 0:
        prev_section = SECTIONS_ORDER[current_index - 1]
        return redirect(url_for('section_workflow', section_name=prev_section))
    elif direction == 'overview':
        return redirect(url_for('grant_overview'))
    
    return redirect(url_for('section_workflow', section_name=SECTIONS_ORDER[current_index]))

@app.route('/overview')
def grant_overview():
    """Show grant overview and progress"""
    grant_overview = get_session_data('grant_overview')
    if not grant_overview:
        return redirect(url_for('index'))
    
    completed_sections = get_session_data('completed_sections') or {}
    processed_docs = get_session_data('processed_docs') or []
    
    return render_template('grant_overview.html',
                         grant_overview=grant_overview,
                         completed_sections=completed_sections,
                         processed_docs=processed_docs,
                         sections_order=SECTIONS_ORDER)

@app.route('/finalize')
def finalize():
    """Final review and PDF generation"""
    grant_overview = get_session_data('grant_overview')
    if not grant_overview:
        return redirect(url_for('index'))
    
    completed_sections = get_session_data('completed_sections') or {}
    
    # Check if all sections are completed
    missing_sections = [s for s in SECTIONS_ORDER if s not in completed_sections]
    
    return render_template('enhanced_finalize.html',
                         completed_sections=completed_sections,
                         missing_sections=missing_sections,
                         sections_order=SECTIONS_ORDER,
                         grant_overview=grant_overview)

@app.route('/download')
def download_pdf():
    """Generate and download PDF of completed grant"""
    grant_overview = get_session_data('grant_overview')
    if not grant_overview:
        return redirect(url_for('index'))
    
    completed_sections = get_session_data('completed_sections') or {}
    
    # Create PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title = Paragraph("Grant Proposal", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Grant Overview
    overview_title = Paragraph("Grant Overview", styles['Heading1'])
    story.append(overview_title)
    overview_text = Paragraph(grant_overview, styles['Normal'])
    story.append(overview_text)
    story.append(Spacer(1, 12))
    
    # Sections
    for section_name in SECTIONS_ORDER:
        if section_name in completed_sections:
            section_title = Paragraph(section_name, styles['Heading1'])
            story.append(section_title)
            section_text = Paragraph(completed_sections[section_name], styles['Normal'])
            story.append(section_text)
            story.append(Spacer(1, 12))
    
    doc.build(story)
    buffer.seek(0)
    
    return send_file(buffer, as_attachment=True, download_name='grant_proposal.pdf', mimetype='application/pdf')

@app.route('/api/section-quality/<section_name>')
def get_section_quality(section_name):
    """API endpoint to get section quality assessment"""
    section_data = get_session_data(f'section_{section_name}') or {}
    return jsonify(section_data)

@app.route('/clear-session')
def clear_session():
    """Clear session and start over"""
    # Clear file-based session data
    session_id = session.get('session_id')
    if session_id:
        session_manager.delete_session(session_id)
    
    # Clear Flask session
    session.clear()
    
    # Clean up uploaded files
    try:
        for file in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, file))
    except:
        pass
    
    flash('Session cleared. Starting fresh.', 'info')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 