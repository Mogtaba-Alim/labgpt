# LabGPT Data Generation App

A web application for generating fine-tuning datasets from GitHub repositories and research papers, designed for creating AI training data for laboratory research contexts.

## Overview

This application provides a user-friendly web interface to:
1. **Process GitHub Repositories**: Clone and analyze code repositories to generate diverse training data including Q&A pairs, code completion tasks, debugging challenges, refactoring suggestions, and docstring generation.
2. **Process Research Papers**: Extract text from PDF research papers and generate question-answer pairs.
3. **Combine Datasets**: Merge code and research paper data into comprehensive training and validation datasets for fine-tuning language models.

## Features

- **Dynamic Repository Input**: Add/remove multiple GitHub repository URLs through an intuitive web interface
- **Research Paper Integration**: Specify a directory containing lab research papers (PDF format)
- **Real-time Processing**: Live progress tracking with detailed logs during data generation
- **Automated Pipeline**: Sequential execution of code analysis and paper processing
- **Professional UI**: Modern, responsive interface similar to the grant generation app
- **Comprehensive Output**: Generates both training and validation datasets in JSON format

## Generated Data Types

### From Code Repositories:
- **Q&A Pairs**: Questions and answers about code functionality, logic, and structure
- **Code Completion**: Partial code with expected completions
- **Debugging Tasks**: Code with potential issues and solutions
- **Refactoring Tasks**: Code improvement suggestions
- **Docstring Generation**: Function signatures with appropriate documentation
- **Dependency Analysis**: Import relationships and project structure

### From Research Papers:
- **Academic Q&A**: Questions about methodology, findings, objectives, tools used
- **Literature Review**: Questions about related work and citations
- **Technical Terms**: Explanations of domain-specific terminology
- **Data Analysis**: Questions about statistical methods and data processing

## Installation

1. **Clone or navigate to the data_generation directory**:
   ```bash
   cd data_generation
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file with your API keys:
   ```
   CLAUDE_API_KEY=your_claude_api_key_here
   OPENAI_KEY=your_openai_api_key_here
   ```

## Usage

1. **Start the Flask application**:
   ```bash
   python app.py
   ```

2. **Access the web interface**:
   Open your browser and go to `http://localhost:5001`

3. **Input your data sources**:
   - **GitHub Repositories**: Add one or more repository URLs (these will be used instead of any hardcoded repositories)
   - **Research Papers Directory**: Provide the full path to your PDF collection (this will be used instead of any hardcoded directory path)

4. **Monitor progress**:
   - Real-time progress tracking
   - Live processing logs
   - Stage-by-stage completion indicators

5. **Access results**:
   - View dataset statistics
   - Download generated JSON files
   - Get usage instructions

## Output Files

The application generates the following files:

- **`combined_dataset_train.json`**: Final training dataset combining code and research data
- **`combined_dataset_val.json`**: Final validation dataset for model evaluation
- **`code_combined_train_dataset.json`**: Intermediate code-only training data
- **`code_combined_val_dataset.json`**: Intermediate code-only validation data

## Dataset Structure

Each dataset entry contains:

```json
{
  "repo": "repository_url",
  "file": "file_path",
  "language": "programming_language",
  "content": "source_code",
  "qa_pairs": [...],
  "completion_tasks": [...],
  "debugging_tasks": [...],
  "refactoring_tasks": [...],
  "docstring_tasks": [...],
  "dependencies": {...},
  "project_dependencies": [...]
}
```

## API Endpoints

- **`GET /`**: Main interface for inputting repositories and papers directory
- **`GET /processing`**: Processing progress page with real-time updates
- **`GET /api/status`**: JSON endpoint for status polling
- **`GET /results`**: Results page showing generated datasets
- **`POST /`**: Start the data generation pipeline

## Requirements

- Python 3.7+
- Flask web framework
- Access to Claude API (Anthropic)
- Access to OpenAI API
- Git (for repository cloning)
- PDF processing capabilities

## File Processing

### Supported Code Files:
- Python (`.py`)
- C++ (`.cpp`)
- R (`.r`, `.R`)

### Supported Paper Formats:
- PDF files (`.pdf`)

## Configuration

The application can be configured by modifying:
- **Port**: Change the port in `app.py` (default: 5001)
- **Processing parameters**: Modify chunk sizes, overlap, and other parameters in the processing scripts
- **Supported file extensions**: Update the `get_code_files()` function in `generateCodeFineTune.py`

## User Input Handling

The application is designed to use **user-provided inputs exclusively**:
- **Repository URLs**: The system uses only the GitHub repository URLs provided through the web interface, not any hardcoded repositories
- **Papers Directory**: The system uses only the research papers directory path provided by the user, not any hardcoded directory paths
- **No Hardcoded Values**: Both `generateCodeFineTune.py` and `createFinalDataOutput.py` have been modified to accept and use user-provided parameters

## Troubleshooting

### Common Issues:
1. **API Key Errors**: Ensure your `.env` file contains valid API keys
2. **Repository Access**: Verify that GitHub repositories are public and accessible
3. **Path Issues**: Use absolute paths for the research papers directory
4. **Memory Issues**: Large repositories may require more system memory

### Log Files:
- Processing logs are displayed in real-time on the web interface
- Check the console output for detailed error messages

## Integration with Fine-Tuning

The generated datasets are compatible with:
- Hugging Face Transformers
- OpenAI fine-tuning APIs
- Custom training pipelines
- Popular machine learning frameworks

Example usage:
```python
import json

# Load the generated dataset
with open('combined_dataset_train.json', 'r') as f:
    training_data = json.load(f)

# Process for your fine-tuning framework
for entry in training_data:
    # Extract relevant training examples
    for qa_pair in entry['qa_pairs']:
        question = qa_pair['question']
        answer = qa_pair['answer']
        # Use in your training pipeline
```

## Contributing

This application is part of the LabGPT suite. For contributions or issues, please refer to the main project documentation.

## License

This project follows the same license as the main LabGPT project. 