# Round 1B: Persona-Driven Document Intelligence

## Overview
Intelligent document analyzer that extracts and prioritizes relevant sections based on user persona and job requirements, building upon Round 1A heading detection.

## Solution Architecture
- **Input**: JSON config with documents, persona, and job task
- **Processing**: Content extraction → TF-IDF relevance scoring → section ranking → subsection analysis  
- **Output**: Ranked sections with detailed subsection analysis

## Dependencies & Models
- **PyMuPDF**: PDF text extraction
- **scikit-learn**: TF-IDF vectorization and cosine similarity
- **NLTK**: Text tokenization (data pre-downloaded)
- **NumPy**: Numerical computations
- **Model Size**: <50MB (lightweight sklearn + NLTK data)

## Build Instructions

### Build Docker Image
docker build --platform linux/amd64 -t persona-analyzer:v1 .

### Run Solution  
docker run --rm
-v $(pwd)/input:/app/input
-v $(pwd)/output:/app/output
--network none
persona-analyzer:v1

## Input Requirements
- Input JSON file with challenge configuration
- PDF documents referenced in the JSON
- Corresponding outline JSON files from Round 1A (same filename as PDF but .json extension)

## Output Format
- `/app/output/result.json` with metadata, extracted_sections, and subsection_analysis

## Performance Specifications
- **Processing Time**: <60 seconds for 5-7 documents
- **Memory Usage**: <1GB RAM
- **Architecture**: CPU-only (AMD64)
- **Network**: Offline execution (no internet calls)

## Key Features
- Builds seamlessly on Round 1A results
- Domain-agnostic content analysis
- Persona-adaptive relevance scoring
- Robust error handling and fallback mechanisms
- Efficient lightweight implementation