# Enhanced LabGPT Grant Generation Pipeline

An intelligent, AI-powered grant writing assistant that leverages your lab's fine-tuned Llama 3.1 8B model and RAG (Retrieval-Augmented Generation) to help generate high-quality grant proposals.

## 🚀 Key Features

### **Two-Stage Generation Pipeline**
1. **Information Extraction**: Intelligently extracts relevant information from uploaded documents for each grant section
2. **Context-Aware Generation**: Combines extracted information with lab-specific RAG context to generate high-quality content

### **Advanced Document Processing**
- **Multi-format Support**: PDF, TXT, DOC, DOCX files
- **OCR Capability**: Handles scanned PDFs with text extraction
- **Intelligent Chunking**: Preserves document structure and context
- **Document Classification**: Automatically identifies grant documents, research papers, and reports

### **Quality Assurance System**
- **Automated Quality Scoring**: AI-based assessment of generated content (1-5 scale)
- **Section-Specific Validation**: Checks for required elements in each section
- **Improvement Suggestions**: Provides actionable feedback for content enhancement
- **Real-time Feedback**: Immediate quality indicators during generation

### **User Experience Enhancements**
- **Progress Tracking**: Visual progress bar and section completion status
- **Section Navigation**: Easy jumping between grant sections
- **Auto-save**: Prevents loss of work with automatic draft saving
- **Responsive Design**: Modern, mobile-friendly interface
- **Keyboard Shortcuts**: Efficient navigation and actions

## 📋 Grant Sections Supported

1. **Background** - Literature review and problem statement
2. **Objectives** - High-level research goals
3. **Specific Aims** - Detailed, testable objectives
4. **Methods** - Experimental procedures and approaches
5. **Preliminary Work** - Prior results and feasibility
6. **Impact/Relevance** - Significance and broader impact
7. **Feasibility, Risks and Mitigation** - Risk assessment and contingency plans
8. **Project Outcomes and Future Directions** - Expected results and follow-up
9. **Research Data Management** - Data handling and sharing plans
10. **Expertise and Resources** - Team qualifications and facilities
11. **Summary of Progress** - PI's track record
12. **Lay Abstract & Summary** - Public-friendly descriptions

## 🔧 Technical Architecture

### **Core Components**

1. **Enhanced Document Processor** (`enhanced_document_processor.py`)
   - Multi-format document parsing
   - OCR for scanned documents
   - Intelligent section-aware chunking
   - Document summarization
   - Content classification

2. **Enhanced Grant Service** (`enhanced_grant_service.py`)
   - Two-stage generation pipeline
   - Quality assessment algorithms
   - Section-specific prompt engineering
   - Feedback incorporation system

3. **Enhanced Flask Application** (`enhanced_app.py`)
   - Modern web interface
   - Session management
   - Progress tracking
   - PDF generation

4. **RAG Service Integration** (`rag_service.py`)
   - Lab-specific knowledge retrieval
   - FAISS vector database
   - Contextual information injection

## 🛠️ Installation & Setup

### **Prerequisites**
- Python 3.8+
- CUDA-capable GPU (recommended for optimal performance)
- Tesseract OCR for scanned document processing

### **Installation Steps**

1. **Clone and Navigate**
   ```bash
   cd grant_generation
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR**
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

4. **Setup Environment Variables**
   ```bash
   export SECRET_KEY="your-secure-secret-key"
   export CUDA_VISIBLE_DEVICES=0  # If using GPU
   ```

5. **Verify Lab Documents Index**
   Ensure your FAISS index and lab documents are properly set up:
   ```
   faiss_index.bin
   chunks.npy
   embeddings.npy
   ```

## 🚀 Usage Guide

### **Starting the Application**
```bash
python enhanced_app.py
```
Access the application at `http://localhost:5000`

### **Grant Generation Workflow**

1. **Home Page Setup**
   - Write a detailed grant overview (200+ characters recommended)
   - Upload supporting documents (optional but recommended)
   - Documents are automatically processed and analyzed

2. **Section-by-Section Generation**
   - Navigate through 12 grant sections in logical order
   - Click "Generate Section" to create initial content
   - Review quality score and AI suggestions
   - Use "Refine with Feedback" to improve content
   - Edit content directly in the text area
   - Save and continue to next section

3. **Quality Control**
   - Monitor quality scores (1-5 scale)
   - Review AI suggestions for improvements
   - Use extracted information and lab context tabs for reference

4. **Final Review and Export**
   - Review completed sections in overview
   - Generate final PDF document
   - Download professional grant proposal

### **Tips for Best Results**

1. **Grant Overview Quality**
   - Provide detailed, specific descriptions
   - Include research questions, methodology, and expected outcomes
   - Mention specific lab capabilities and expertise

2. **Document Upload Strategy**
   - Include previous successful grants from your lab
   - Add relevant research papers and preliminary data
   - Upload CV/biosketch information for PI sections

3. **Iterative Improvement**
   - Use the refinement feature with specific feedback
   - Review quality scores and implement suggestions
   - Cross-reference with lab context for consistency

## 🎯 Advanced Features

### **Context Length Management**
The system automatically handles large documents by:
- Intelligent chunking that preserves context
- Hierarchical summarization for long documents
- Dynamic context window optimization
- Relevant information extraction per section

### **Quality Assessment Metrics**
- **Length Analysis**: Appropriate section length
- **Content Relevance**: Section-specific requirements
- **Clarity Indicators**: Sentence structure and flow
- **Completeness Checks**: Required elements present

### **Prompt Engineering**
- **Section-Specific Prompts**: Tailored for each grant section
- **Two-Stage Prompting**: Extraction + Generation
- **Context Integration**: Seamless information weaving
- **Feedback Incorporation**: Iterative improvement prompts

## 🔍 Troubleshooting

### **Common Issues**

1. **Model Loading Errors**
   - Ensure sufficient GPU memory (8GB+ recommended)
   - Try CPU-only mode if GPU unavailable
   - Check internet connection for model downloads

2. **Document Processing Issues**
   - Verify Tesseract installation for OCR
   - Check file permissions for uploads
   - Ensure supported file formats

3. **Generation Quality Issues**
   - Provide more detailed grant overview
   - Upload more relevant supporting documents
   - Use refinement feature with specific feedback

### **Performance Optimization**

1. **Memory Management**
   - Use quantization options in model loading
   - Implement batch processing for large documents
   - Monitor system resources

2. **Speed Improvements**
   - Enable GPU acceleration
   - Use model caching
   - Optimize chunk sizes

## 📊 Performance Metrics

Expected performance on modern hardware:
- **Document Processing**: 1-2 minutes for typical PDF
- **Section Generation**: 30-60 seconds per section
- **Quality Assessment**: Near real-time
- **PDF Export**: 5-10 seconds

## 🤝 Contributing

To contribute improvements:
1. Fork the repository
2. Create feature branch
3. Implement enhancements
4. Add tests and documentation
5. Submit pull request

## 📄 License

This project is part of the LABGPT research initiative. Please respect academic use guidelines and cite appropriately.

## 🆘 Support

For technical support or questions:
- Check troubleshooting section above
- Review log files for error details
- Ensure all dependencies are properly installed

## 🔮 Future Enhancements

Planned improvements include:
- Multi-language support
- Advanced citation management
- Collaborative editing features
- Integration with funding agency templates
- Enhanced visualization tools
- Advanced analytics and insights 