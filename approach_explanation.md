# Persona-Driven Document Intelligence Solution

## Methodology Overview

This solution builds upon Round 1A's heading detection to create an intelligent document analyst that extracts and prioritizes content based on user persona and specific job requirements.

## Core Algorithm

### 1. Input Processing
- Reads JSON configuration containing documents, persona role, and job task
- Loads PDF documents and their corresponding outline files from Round 1A
- Combines persona role and job task into a search query for relevance matching

### 2. Content Extraction
- Uses Round 1A outline data to extract structured content sections
- Maintains page number mapping for precise location tracking  
- Determines section boundaries using heading hierarchy and page positions
- Combines section titles with content for comprehensive analysis

### 3. Relevance Scoring
- Employs TF-IDF vectorization to convert text and persona queries into numerical representations
- Calculates cosine similarity between section content and persona requirements
- Implements fallback scoring based on heading levels and content length for robustness
- Ranks sections by relevance score and assigns importance rankings

### 4. Subsection Analysis
- Extracts granular content from top-ranked sections using sentence tokenization
- Scores individual sentences against persona query for targeted content extraction
- Selects most relevant sentences while maintaining coherence and context
- Limits output to substantial content (>50 characters) to ensure quality

## Technical Implementation

### Robustness Features
- **Error Handling**: Graceful degradation when NLTK or TF-IDF processing fails
- **Flexible Input**: Automatically detects input JSON files regardless of naming
- **Missing File Handling**: Continues processing even if some documents are unavailable
- **Content Validation**: Ensures extracted content meets minimum quality thresholds

### Performance Optimizations
- **Lightweight Models**: Uses sklearn TF-IDF instead of heavy language models
- **Efficient Processing**: Sequential document processing to manage memory usage
- **Smart Fallbacks**: Multiple scoring strategies to ensure consistent results

### Domain Adaptability
- **Generic Approach**: No hardcoded domain-specific logic
- **Persona-Adaptive**: Dynamically adjusts scoring based on different user roles
- **Task-Flexible**: Accommodates various job-to-be-done requirements

This approach ensures high relevance accuracy across diverse document types and user requirements while maintaining fast processing and broad applicability.