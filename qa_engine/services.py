import faiss
import numpy as np
from documents.models import OCRLine

# Try to import sentence_transformers, but allow graceful degradation if unavailable
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, OSError) as e:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

def get_all_texts():
    """
    Retrieve all OCR line texts and their IDs.
    """
    lines = OCRLine.objects.select_related('page__document').all()
    texts = [line.text for line in lines]
    ids = [line.id for line in lines]
    return texts, ids

def build_index():
    """
    Build FAISS index for all OCR texts.
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise OSError("Sentence Transformers not available. PyTorch compatibility issue - please use Python 3.8-3.12.")
    
    # Load the embedding model (multilingual for Tamil/English)
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    texts, ids = get_all_texts()
    if not texts:
        return None, None, None
    embeddings = model.encode(texts, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index, embeddings, ids

def search_relevant_texts(question, top_k=5):
    """
    Search for the most relevant OCR lines to the question.
    Returns list of (OCRLine, score)
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return []
    
    index, embeddings, ids = build_index()
    if index is None:
        return []
    
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    query_embedding = model.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:
            line_id = ids[idx]
            line = OCRLine.objects.get(id=line_id)
            results.append((line, dist))
    return results
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:
            line_id = ids[idx]
            line = OCRLine.objects.get(id=line_id)
            results.append((line, dist))
    return results

def clean_text(text):
    """
    Clean OCR text by removing junk characters and normalizing.
    """
    # Remove non-alphanumeric characters except spaces
    cleaned = ''.join(c for c in text if c.isalnum() or c.isspace())
    # Normalize spaces
    cleaned = ' '.join(cleaned.split())
    return cleaned.lower()

def find_relevant_lines(question, top_k=5):
    """
    Find relevant OCR lines using improved keyword matching.
    Returns list of (OCRLine, score) sorted by score desc, then length desc.
    """
    print(f"DEBUG: Finding relevant lines for question: '{question}'")
    lines = OCRLine.objects.select_related('page__document').all()
    if not lines:
        print("DEBUG: No OCR lines found in database")
        return []
    
    # Clean and tokenize question
    question_clean = clean_text(question)
    stop_words = {'define', 'what', 'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'who', 'which', 'when', 'where', 'why', 'how'}
    question_words = [word for word in question_clean.split() if len(word) >= 3 and word not in stop_words]
    print(f"DEBUG: Cleaned question keywords: {question_words}")
    
    results = []
    for line in lines:
        if len(line.text.strip()) < 40:
            continue  # Skip short lines
        text_clean = clean_text(line.text)
        score = sum(1 for word in question_words if word in text_clean)
        if score >= 2:  # Minimum score
            results.append((line, score))
    
    # Sort by score descending, then by text length descending
    results.sort(key=lambda x: (-x[1], -len(x[0].text)))
    print(f"DEBUG: Found {len(results)} relevant lines, top {min(top_k, len(results))} with scores: {[(line.id, score) for line, score in results[:top_k]]}")
    return results[:top_k]

def extract_answer(question, text):
    """
    Extract a short, direct answer from the text based on the question type.
    """
    question_lower = question.lower()
    text_lower = text.lower()
    
    # Define question patterns
    if question_lower.startswith(('who ', 'who\'s ', 'whose ')):
        # Look for patterns like "X is", "X owns", "X was"
        patterns = [' is ', ' was ', ' owns ', ' belongs to ', ' by ']
        for pattern in patterns:
            if pattern in text_lower:
                parts = text.split(pattern, 1)
                if len(parts) > 1:
                    answer = parts[1].strip().split('.')[0].split(',')[0].strip()
                    return answer[:100]  # Limit length
    
    elif question_lower.startswith(('what is ', 'what\'s ', 'what are ')):
        # Look for "is" or "are" followed by the answer
        if ' is ' in text_lower:
            parts = text.split(' is ', 1)
            if len(parts) > 1:
                answer = parts[1].strip().split('.')[0].split(',')[0].strip()
                return answer[:100]
        elif ' are ' in text_lower:
            parts = text.split(' are ', 1)
            if len(parts) > 1:
                answer = parts[1].strip().split('.')[0].split(',')[0].strip()
                return answer[:100]
    
    elif question_lower.startswith(('which ', 'where ', 'when ')):
        # Similar to what is
        question_word = question_lower.split()[0]
        if f' {question_word} ' in text_lower:
            parts = text.split(f' {question_word} ', 1)
            if len(parts) > 1:
                answer = parts[1].strip().split('.')[0].split(',')[0].strip()
                return answer[:100]
    
    # Fallback: return first meaningful phrase
    sentences = text.split('.')
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10 and len(sentence) < 200:
            return sentence[:100]
    
    # If no good extraction, return a short version of the text
    return text[:100] + ('...' if len(text) > 100 else '')

def generate_answer(question):
    """
    Generate answer using improved rule-based logic without ML dependencies.
    Returns answer_text, page_number, line_number, line_obj
    """
    print(f"DEBUG: generate_answer called with question: '{question}'")
    results = find_relevant_lines(question, top_k=1)
    if results:
        line, score = results[0]
        print(f"DEBUG: Selected line ID: {line.id}, Score: {score}, Text length: {len(line.text)}, Text: '{line.text[:100]}...'")
        answer = extract_answer(question, line.text)
        print(f"DEBUG: Extracted answer: '{answer}'")
        return answer, line.page.page_number, line.line_number, line
    else:
        print("DEBUG: No relevant lines found with score >= 2")
        return "No relevant information found.", None, None, None