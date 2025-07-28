import os
import json
import fitz  # PyMuPDF
import re
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

class PersonaDrivenAnalyzer:
    def __init__(self):
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Extract text content page by page"""
        doc = fitz.open(pdf_path)
        page_texts = {}
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            page_texts[page_num + 1] = text
            
        doc.close()
        return page_texts

    def get_section_content(self, page_texts: Dict[int, str], outline: List[Dict], doc_filename: str) -> List[Dict]:
        """Extract content for each section based on outline"""
        sections_with_content = []
        
        for i, section in enumerate(outline):
            start_page = section['page']
            
            # Determine end page (next section's page or last page)
            if i + 1 < len(outline):
                end_page = outline[i + 1]['page'] - 1
            else:
                end_page = max(page_texts.keys())
            
            # Extract content from start_page to end_page
            content = ""
            for page_num in range(start_page, end_page + 1):
                if page_num in page_texts:
                    page_text = page_texts[page_num]
                    
                    # Find section start in the page
                    if page_num == start_page:
                        section_start = page_text.find(section['text'])
                        if section_start != -1:
                            content += page_text[section_start:]
                        else:
                            content += page_text
                    else:
                        content += page_text
            
            sections_with_content.append({
                'document': doc_filename,
                'section_title': section['text'],
                'page_number': section['page'],
                'level': section['level'],
                'content': content.strip()
            })
        
        return sections_with_content

    def create_persona_query(self, persona_role: str, job_task: str) -> str:
        """Create search query from persona and job"""
        return f"{persona_role} {job_task}".lower()

    def score_section_relevance(self, sections: List[Dict], persona_query: str) -> List[Dict]:
        """Score sections based on relevance to persona and job"""
        if not sections:
            return sections
            
        # Prepare texts for vectorization
        section_texts = []
        for section in sections:
            # Combine section title and content for better matching
            combined_text = f"{section['section_title']} {section['content']}"
            section_texts.append(combined_text)
        
        # Add persona query
        all_texts = section_texts + [persona_query]
        
        try:
            # Vectorize
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Calculate similarity with persona query
            persona_vector = tfidf_matrix[-1]
            section_vectors = tfidf_matrix[:-1]
            
            similarities = cosine_similarity(section_vectors, persona_vector.reshape(1, -1)).flatten()
            
            # Add scores to sections
            for i, section in enumerate(sections):
                section['relevance_score'] = float(similarities[i])
        
        except Exception as e:
            # Fallback: assign scores based on section level and length
            for i, section in enumerate(sections):
                level_score = {'H1': 0.8, 'H2': 0.6, 'H3': 0.4}.get(section['level'], 0.2)
                content_score = min(len(section['content']) / 1000, 0.5)
                section['relevance_score'] = level_score + content_score
        
        # Sort by relevance score
        sections_sorted = sorted(sections, key=lambda x: x['relevance_score'], reverse=True)
        
        # Add importance rank
        for i, section in enumerate(sections_sorted):
            section['importance_rank'] = i + 1
        
        return sections_sorted

    def extract_top_subsections(self, sections: List[Dict], persona_query: str, max_subsections: int = 5) -> List[Dict]:
        """Extract relevant subsections from top sections"""
        subsections = []
        
        # Take top 3 sections for subsection analysis
        top_sections = sections[:3]
        
        for section in top_sections:
            content = section['content']
            if not content.strip():
                continue
                
            # Split into sentences
            try:
                sentences = sent_tokenize(content)
            except:
                # Fallback if NLTK fails
                sentences = content.split('. ')
            
            if len(sentences) <= 1:
                # If very short content, use the whole thing
                refined_text = content.strip()
                if len(refined_text) > 50:  # Only add if substantial
                    subsections.append({
                        'document': section['document'],
                        'refined_text': refined_text,
                        'page_number': section['page_number']
                    })
                continue
            
            # For longer content, extract key sentences
            try:
                # Score sentences against persona query
                sentence_texts = sentences + [persona_query]
                tfidf_matrix = self.vectorizer.fit_transform(sentence_texts)
                
                persona_vector = tfidf_matrix[-1]
                sentence_vectors = tfidf_matrix[:-1]
                
                similarities = cosine_similarity(sentence_vectors, persona_vector.reshape(1, -1)).flatten()
                
                # Get top 2-3 sentences per section
                top_indices = np.argsort(similarities)[-2:][::-1]
                
                selected_sentences = [sentences[idx].strip() for idx in top_indices if sentences[idx].strip()]
                refined_text = ' '.join(selected_sentences)
                
            except:
                # Fallback: take first few sentences
                refined_text = '. '.join(sentences[:2]).strip()
            
            if len(refined_text) > 50:  # Only add substantial content
                subsections.append({
                    'document': section['document'],
                    'refined_text': refined_text,
                    'page_number': section['page_number']
                })
        
        return subsections[:max_subsections]

    def process_documents(self, input_dir: str, output_dir: str):
        """Main processing function"""
        # Read input configuration
        input_files = [f for f in os.listdir(input_dir) if f.endswith('.json') and 'input' in f.lower()]
        if not input_files:
            # Look for any JSON file
            input_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        
        if not input_files:
            raise FileNotFoundError("No input JSON file found")
        
        config_path = os.path.join(input_dir, input_files[0])
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Extract configuration
        documents = [doc['filename'] for doc in config['documents']]
        persona_role = config['persona']['role']
        job_task = config['job_to_be_done']['task']
        
        # Create persona query
        persona_query = self.create_persona_query(persona_role, job_task)
        
        all_sections = []
        
        # Process each document
        for doc_filename in documents:
            pdf_path = os.path.join(input_dir, doc_filename)
            outline_path = os.path.join(input_dir, doc_filename.replace('.pdf', '.json'))
            
            if not os.path.exists(pdf_path):
                print(f"Warning: PDF file {doc_filename} not found, skipping...")
                continue
                
            if not os.path.exists(outline_path):
                print(f"Warning: Outline file for {doc_filename} not found, skipping...")
                continue
            
            # Load outline from 1A
            try:
                with open(outline_path, 'r') as f:
                    outline_data = json.load(f)
                
                # Extract page texts
                page_texts = self.extract_text_from_pdf(pdf_path)
                
                # Get section content
                sections_with_content = self.get_section_content(page_texts, outline_data['outline'], doc_filename)
                all_sections.extend(sections_with_content)
                
            except Exception as e:
                print(f"Error processing {doc_filename}: {e}")
                continue
        
        if not all_sections:
            raise Exception("No sections could be extracted from any document")
        
        # Score and rank all sections
        ranked_sections = self.score_section_relevance(all_sections, persona_query)
        
        # Take top 5 sections for output
        top_sections = ranked_sections[:5]
        
        # Extract subsections
        subsection_analysis = self.extract_top_subsections(ranked_sections, persona_query)
        
        # Prepare output in the exact format required
        output = {
            "metadata": {
                "input_documents": documents,
                "persona": persona_role,
                "job_to_be_done": job_task,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [
                {
                    "document": section['document'],
                    "section_title": section['section_title'],
                    "importance_rank": section['importance_rank'],
                    "page_number": section['page_number']
                }
                for section in top_sections
            ],
            "subsection_analysis": subsection_analysis
        }
        
        # Save output
        output_path = os.path.join(output_dir, 'result.json')
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Processing complete. Results saved to {output_path}")

if __name__ == "__main__":
    analyzer = PersonaDrivenAnalyzer()
    analyzer.process_documents('/app/input', '/app/output')
