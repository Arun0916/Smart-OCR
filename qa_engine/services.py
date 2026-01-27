import faiss
import numpy as np
import re
from documents.models import OCRLine

# Try to import sentence_transformers, but allow graceful degradation if unavailable
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, OSError) as e:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

# Cache for the model to avoid reloading
_MODEL_CACHE = None

def _get_cached_model():
    """
    Get or create cached SentenceTransformer model.
    Internal helper function.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is None and SENTENCE_TRANSFORMERS_AVAILABLE:
        _MODEL_CACHE = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _MODEL_CACHE

def normalize_text(text):
    """
    Normalize OCR text: lowercase, remove extra spaces, fix broken words.
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Fix broken words: combine short words with following lowercase words
    words = text.split()
    fixed = []
    i = 0
    while i < len(words):
        # Check bounds and ensure words[i+1] is not empty before accessing first character
        if (i < len(words) - 1 and 
            len(words[i]) <= 3 and 
            words[i+1] and 
            len(words[i+1]) > 0 and 
            words[i+1][0].islower()):
            fixed.append(words[i] + words[i+1])
            i += 2
        else:
            fixed.append(words[i])
            i += 1
    return ' '.join(fixed)

def classify_question(question):
    """
    Classify the question into specific types for targeted extraction.
    """
    q = question.lower()
    if any(kw in q for kw in ['account', 'a/c', 'acc', 'sb', 'ac', 'acct', 'ifsc', 'cif', 'branch code']):
        return 'ACCOUNT_NUMBER'
    elif any(kw in q for kw in ['email', 'mail', 'e-mail', 'contact']):
        return 'EMAIL'
    elif any(kw in q for kw in ['mobile', 'phone', 'cell', 'contact']):
        return 'MOBILE'
    elif any(kw in q for kw in ['name', 'applicant', 'filled', 'who']):
        return 'NAME'
    elif any(kw in q for kw in ['address', 'addr', 'location']):
        return 'ADDRESS'
    else:
        return 'GENERAL'

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
    Returns (index, embeddings, ids) or (None, None, None) if unsuccessful.
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise OSError("Sentence Transformers not available. PyTorch compatibility issue - please use Python 3.8-3.12.")
    
    # Use cached model
    model = _get_cached_model()
    if model is None:
        return None, None, None
    
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
    if index is None or embeddings is None or ids is None:
        return []
    
    # Use cached model
    model = _get_cached_model()
    if model is None:
        return []
    
    query_embedding = model.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:
            line_id = ids[idx]
            try:
                line = OCRLine.objects.get(id=line_id)
                results.append((line, dist))
            except OCRLine.DoesNotExist:
                continue
    
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

def fuzzy_keyword_match(text, keywords):
    """
    Fuzzy match keywords in text to handle OCR errors.
    Returns True if any keyword matches (with tolerance for 1-2 char differences).
    """
    text_lower = text.lower()
    for keyword in keywords:
        # Exact match
        if keyword in text_lower:
            return True
        # Fuzzy match: check if keyword with 1 char missing/changed exists
        if len(keyword) >= 4:
            # Check substring match (e.g., "emai" matches "email")
            if keyword[:-1] in text_lower or keyword[1:] in text_lower:
                return True
            # Check character variations
            for variant in [keyword[:-1], keyword[1:], keyword[:-2]]:
                if variant in text_lower:
                    return True
    return False

def find_relevant_lines(question, top_k=5):
    """
    Find relevant OCR lines using improved keyword matching with fuzzy matching.
    Returns list of (full_line_text, OCRLine, score) sorted by score desc.
    """
    print(f"DEBUG: Finding relevant lines for question: '{question}'")
    lines = OCRLine.objects.select_related('page__document').all()
    if not lines.exists():
        print("DEBUG: No OCR lines found in database")
        return []
    
    # Clean and tokenize question
    question_clean = clean_text(question)
    stop_words = {
        'define', 'what', 'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 
        'at', 'to', 'for', 'of', 'with', 'by', 'are', 'was', 'were', 'be', 'been', 
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
        'could', 'should', 'may', 'might', 'must', 'can', 'shall', 'this', 'that', 
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 
        'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 
        'who', 'which', 'when', 'where', 'why', 'how'
    }
    question_words = [word for word in question_clean.split() if len(word) >= 3 and word not in stop_words]
    
    # Add synonyms for better matching
    synonyms = {
        'account': ['account', 'a/c', 'acc', 'sb', 'ac', 'acct'],
        'email': ['email', 'mail', 'e-mail', 'emai', 'contact'],  # Added 'emai' for OCR errors
        'mobile': ['mobile', 'phone', 'cell', 'contact', 'naber', 'phon'],  # Added OCR error variants
        'name': ['name', 'applicant', 'filled', 'who'],
        'address': ['address', 'addr', 'location', 'addres', 'postal', 'post', 'paste'],  # Added OCR variants
        'ifsc': ['ifsc', 'branch code', 'fsc'],
        'bank': ['bank', 'branch', 'sbi', 'hdfc', 'icici'],
        'what': ['what', 'about', 'details', 'description'],
    }
    
    expanded_words = set(question_words)
    for word in question_words:
        if word in synonyms:
            expanded_words.update(synonyms[word])
    question_words = list(expanded_words)
    print(f"DEBUG: Expanded question keywords: {question_words}")
    
    # Get question type for targeted keyword boosting
    question_type = classify_question(question)
    
    results = []
    for line in lines:
        if len(line.text.strip()) < 10:
            continue  # Skip very short lines
        
        # Use the full line text
        text_clean = clean_text(line.text)
        normalized_text = normalize_text(line.text)
        
        # Calculate score based on keyword matches (with fuzzy matching)
        score = 0
        for word in question_words:
            # Exact match
            if word in normalized_text:
                score += 1
            # Fuzzy match for longer words
            elif len(word) >= 4 and (word[:-1] in normalized_text or word[1:] in normalized_text):
                score += 0.5
        
        # Special boosting based on question type with fuzzy matching
        if question_type == 'ACCOUNT_NUMBER':
            boost_keywords = ['account', 'a/c', 'acc', 'sb', 'ifsc', 'cif', 'branch']
            if fuzzy_keyword_match(normalized_text, boost_keywords):
                score += 2
        elif question_type == 'EMAIL':
            boost_keywords = ['email', 'emai', 'mail', 'e-mail', 'contact', '@']
            if fuzzy_keyword_match(normalized_text, boost_keywords) or '@' in line.text:
                score += 2
        elif question_type == 'MOBILE':
            boost_keywords = ['mobile', 'phone', 'cell', 'contact', 'naber', 'phon']
            if fuzzy_keyword_match(normalized_text, boost_keywords):
                score += 2
            # Also boost if we find 10-digit patterns
            if re.search(r'\d{10}', normalized_text.replace(' ', '')):
                score += 1
        elif question_type == 'NAME':
            boost_keywords = ['name', 'applicant', 'filled', 'who']
            if fuzzy_keyword_match(normalized_text, boost_keywords):
                score += 2
        elif question_type == 'ADDRESS':
            boost_keywords = ['address', 'addres', 'addr', 'location', 'postal', 'post', 'paste', 'nagar', 'street', 'road']
            if fuzzy_keyword_match(normalized_text, boost_keywords):
                score += 2
        
        if score >= 0.5:  # Lower threshold to catch more potential matches
            # Store the full original line text
            results.append((line.text, line, score))
    
    # Sort by score descending
    results.sort(key=lambda x: -x[2])
    print(f"DEBUG: Found {len(results)} relevant lines, top {min(top_k, len(results))} with scores: {[(line.id, score) for text, line, score in results[:top_k]]}")
    return results[:top_k]

def extract_answer(question, text):
    """
    Extract a precise answer from normalized text based on question type.
    Uses regex for structured data, pattern matching for names/addresses.
    Returns the answer string or None if not extractable.
    """
    question_type = classify_question(question)
    text_normalized = normalize_text(text)  # Normalize the input text
    
    print(f"DEBUG extract_answer: question_type={question_type}, text_length={len(text)}")
    print(f"DEBUG extract_answer: first 200 chars: '{text_normalized[:200]}'")
    
    if question_type == 'ACCOUNT_NUMBER':
        # First try: Look for any sequence of 9-18 digits (most common account number format)
        # This handles cases where digits might have spaces in OCR
        text_no_spaces = text_normalized.replace(' ', '')
        digit_sequences = re.findall(r'\d{9,18}', text_no_spaces)
        if digit_sequences:
            # Return the first valid sequence
            for seq in digit_sequences:
                print(f"DEBUG extract_answer: Found digit sequence: {seq}")
                return seq
        
        # Second try: Extract account number with labels
        patterns = [
            r'account\s*(?:no|number|num)?\s*[:\-]?\s*([\d\s\-]{9,25})',
            r'a/?c\s*(?:no|number|num)?\s*[:\-]?\s*([\d\s\-]{9,25})',
            r'acc\s*(?:no|number|num)?\s*[:\-]?\s*([\d\s\-]{9,25})',
            r'sb\s*a/?c\s*[:\-]?\s*([\d\s\-]{9,25})',
            r'acct\s*(?:no|number|num)?\s*[:\-]?\s*([\d\s\-]{9,25})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text_normalized, re.IGNORECASE)
            if match:
                account = match.group(1).replace(' ', '').replace('-', '')
                if account.isdigit() and 9 <= len(account) <= 18:
                    print(f"DEBUG extract_answer: Extracted account number: {account}")
                    return account
        
        # IFSC code
        match = re.search(r'(?:ifsc|fsc)\s*(?:code)?\s*[:\-]?\s*([A-Z0-9]{11})', text_normalized.upper(), re.IGNORECASE)
        if match:
            ifsc = match.group(1).upper()
            print(f"DEBUG extract_answer: Extracted IFSC: {ifsc}")
            return ifsc
        
        return None
    
    elif question_type == 'EMAIL':
        # Extract valid email address (handles OCR errors)
        # Look for @ symbol and extract email around it
        if '@' in text:
            match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text, re.IGNORECASE)
            if match:
                email = match.group(0).lower()
                print(f"DEBUG extract_answer: Extracted email: {email}")
                return email
        
        # Try with common OCR errors: emai, mai, etc.
        patterns = [
            r'(?:email|emai|mail)\s*[:\-]?\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                email = match.group(1).lower()
                print(f"DEBUG extract_answer: Extracted email from pattern: {email}")
                return email
        
        return None
    
    elif question_type == 'MOBILE':
        # Extract Indian mobile numbers: 10 digits, starting with 6-9
        # First remove all spaces to handle OCR errors like "98765 43210"
        text_no_spaces = text_normalized.replace(' ', '').replace('-', '')
        
        # Look for 10-digit numbers starting with 6-9
        matches = re.findall(r'[6-9]\d{9}', text_no_spaces)
        if matches:
            mobile = matches[0]
            print(f"DEBUG extract_answer: Extracted mobile: {mobile}")
            return mobile
        
        # Try with labeled patterns (with OCR error tolerance)
        patterns = [
            r'(?:mobile|phone|cell|naber|phon)\s*(?:no|number|num)?\s*[:\-]?\s*([\d\s\-]{10,15})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text_normalized, re.IGNORECASE)
            if match:
                mobile = match.group(1).replace(' ', '').replace('-', '')
                if mobile.isdigit() and len(mobile) == 10 and mobile[0] in '6789':
                    print(f"DEBUG extract_answer: Extracted mobile from pattern: {mobile}")
                    return mobile
        
        return None
    
    elif question_type == 'NAME':
        # Extract applicant name or person who filled the form
        patterns = [
            r'(?:applicant|customer)\s*(?:name)?\s*[:\-]?\s*([A-Za-z][A-Za-z\s]{2,49})',
            r'name\s*[:\-]?\s*([A-Za-z][A-Za-z\s]{2,49})',
            r'filled\s*by\s*[:\-]?\s*([A-Za-z][A-Za-z\s]{2,49})',
            r'(?:fa|father)\s*name\s*[:\-]?\s*([A-Za-z][A-Za-z\s]{2,49})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text_normalized, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Clean: remove extra spaces, limit length, ensure alphabetic
                name = re.sub(r'\s+', ' ', name)
                name = name[:50]
                # Check if mostly alphabetic (allow some spaces)
                if re.match(r'^[A-Za-z\s]+$', name) and len(name.split()) <= 5:
                    cleaned_name = name.title()
                    print(f"DEBUG extract_answer: Extracted name: {cleaned_name}")
                    return cleaned_name
        return None
    
    elif question_type == 'ADDRESS':
        # Extract address: text after address keyword (with OCR error tolerance)
        patterns = [
            r'(?:address|addres|addr)\s*[:\-]?\s*([A-Za-z0-9\s,./\-]+?)(?:\s*(?:pin|zip|code|ifsc|account|mobile|email|phone)|\s*$)',
            r'(?:postal|post)\s*(?:al)?\s*(?:address|addr)?\s*[:\-]?\s*([A-Za-z0-9\s,./\-]+?)(?:\s*(?:pin|zip|code|ifsc|account|mobile|email|phone)|\s*$)',
            r'paste\s*ro\s*[:\-]?\s*([A-Za-z0-9\s,./\-]+?)(?:\s*(?:pin|zip|code|ifsc|account|mobile|email|phone)|\s*$)',  # OCR error for "postal"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_normalized, re.IGNORECASE)
            if match:
                address = match.group(1).strip()
                # Clean and limit
                address = re.sub(r'\s+', ' ', address)
                if len(address) > 10:  # Must be substantial
                    cleaned_address = address[:200]
                    print(f"DEBUG extract_answer: Extracted address: {cleaned_address}")
                    return cleaned_address
        
        # Fallback: if we find typical address components (numbers + location names)
        if re.search(r'\d+\s*[a-z]+\s+(?:nagar|street|road|avenue|lane)', text_normalized, re.IGNORECASE):
            # Extract the portion containing address-like content
            match = re.search(r'(\d+[a-z\s,./\-]+(?:nagar|street|road|avenue|lane)[a-z\s,./\-]+\d{6})', text_normalized, re.IGNORECASE)
            if match:
                address = match.group(1).strip()
                address = re.sub(r'\s+', ' ', address)
                print(f"DEBUG extract_answer: Extracted address (fallback): {address}")
                return address[:200]
        
        return None
    
    else:  # GENERAL
        # Fallback to pattern matching for general questions
        question_lower = question.lower().strip()
        
        # Handle ownership questions
        if any(question_lower.startswith(prefix) for prefix in ['who owns', 'who\'s the owner of', 'whose', 'who is the owner']):
            patterns = [
                r'owned by\s+([^\.,;!?]+)',
                r'owner\s+is\s+([^\.,;!?]+)',
                r'belongs to\s+([^\.,;!?]+)',
                r'is owned by\s+([^\.,;!?]+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, text_normalized, re.IGNORECASE)
                if match:
                    answer = match.group(1).strip()
                    answer = re.sub(r'\s*(?:and|or|but|however|therefore|thus|so|because|since|while|although|despite|after|before|during|when|where|why|how).*$', '', answer, flags=re.IGNORECASE)
                    return answer[:100]
        
        # Handle "what is" questions
        elif any(question_lower.startswith(prefix) for prefix in ['what is', 'what\'s', 'what are']):
            patterns = [
                r'is\s+([^\.,;!?]+)',
                r'are\s+([^\.,;!?]+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, text_normalized, re.IGNORECASE)
                if match:
                    answer = match.group(1).strip()
                    answer = re.sub(r'\s*(?:and|or|but|however|therefore|thus|so|because|since|while|although|despite|after|before|during|when|where|why|how).*$', '', answer, flags=re.IGNORECASE)
                    return answer[:100]
        
        # Fallback: extract phrases with question keywords
        question_words = re.findall(r'\b\w+\b', question)
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall', 'who', 'what', 'which', 'when', 'where', 'why', 'how'])
        key_words = [w for w in question_words if w.lower() not in stop_words and len(w) > 2]
        
        if key_words:
            sentences = re.split(r'[.!?]', text_normalized)
            for sentence in sentences:
                sentence = sentence.strip()
                if any(kw.lower() in sentence.lower() for kw in key_words):
                    words = sentence.split()
                    if len(words) > 3:
                        for i, word in enumerate(words):
                            if any(kw.lower() in word.lower() for kw in key_words):
                                answer = ' '.join(words[i:i+10])
                                return answer[:100]
                    elif len(sentence) > 10:
                        return sentence[:100]
    
    print(f"DEBUG extract_answer: No answer extracted")
    return None

def generate_answer(question):
    """
    Generate precise answer from document OCR text using rule-based extraction.
    Searches relevant lines, extracts based on question type, returns best match.
    
    Example test cases:
    - "What is the account number?" → Extracts 123456789012 (from "account no: 123456789012")
    - "Find email id" → Extracts user@example.com (from "email: user@example.com")
    - "Who filled this form?" → Extracts John Doe (from "applicant name: john doe")
    - "What is the mobile number?" → Extracts 9876543210 (from "mobile: 9876543210")
    - "What is the address?" → Extracts "123 Main St, City" (from "address: 123 main st, city")
    
    Returns answer_text, page_number, line_number, line_obj
    """
    print(f"DEBUG: generate_answer called with question: '{question}'")
    results = find_relevant_lines(question, top_k=10)  # Get top 10 relevant lines
    extracted = False
    
    for full_line_text, line, score in results:
        print(f"DEBUG: Checking line ID: {line.id}, Score: {score}")
        print(f"DEBUG: Full line text: '{full_line_text}'")
        answer = extract_answer(question, full_line_text)
        if answer:
            print(f"DEBUG: Extracted answer: '{answer}'")
            return answer, line.page.page_number, line.line_number, line
        extracted = True  # Relevant content found but extraction failed
    
    if extracted:
        print("DEBUG: Relevant information exists but not clearly readable")
        # Try to get the full text of the most relevant line
        if results:
            best_text, best_line, best_score = results[0]
            print(f"DEBUG: Returning full text from best match: '{best_text[:200]}'")
            return f"Found relevant information: {best_text[:200]}", best_line.page.page_number, best_line.line_number, best_line
        return "The requested information exists but is not clearly readable in the document.", None, None, None
    else:
        # Fallback: search all OCR text for exact keyword matches
        print("DEBUG: No answer from top results, checking all text for exact keywords")
        # Compute question words with synonyms
        question_clean = clean_text(question)
        stop_words = {
            'define', 'what', 'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 
            'at', 'to', 'for', 'of', 'with', 'by', 'are', 'was', 'were', 'be', 'been', 
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
            'could', 'should', 'may', 'might', 'must', 'can', 'shall', 'this', 'that', 
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 
            'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 
            'who', 'which', 'when', 'where', 'why', 'how'
        }
        question_words_base = [word for word in question_clean.split() if len(word) >= 3 and word not in stop_words]
        synonyms = {
            'account': ['account', 'a/c', 'acc', 'sb', 'ac', 'acct'],
            'email': ['email', 'mail', 'e-mail', 'emai', 'contact'],
            'mobile': ['mobile', 'phone', 'cell', 'contact', 'naber'],
            'name': ['name', 'applicant', 'filled', 'who'],
            'address': ['address', 'addr', 'location', 'postal', 'post', 'paste'],
            'ifsc': ['ifsc', 'branch code'],
            'bank': ['bank', 'branch', 'sbi', 'hdfc', 'icici'],
            'what': ['what', 'about', 'details', 'description'],
        }
        question_words = set(question_words_base)
        for word in question_words_base:
            if word in synonyms:
                question_words.update(synonyms[word])
        question_words = list(question_words)
        
        all_lines = OCRLine.objects.select_related('page__document').all()
        for line in all_lines:
            text_clean = clean_text(line.text)
            normalized = normalize_text(line.text)
            
            # Use fuzzy matching in fallback too
            if any(word in normalized for word in question_words) or fuzzy_keyword_match(normalized, question_words):
                answer = extract_answer(question, line.text)
                if answer:
                    print(f"DEBUG: Found answer in full search: '{answer}'")
                    return answer + " (extracted from document)", line.page.page_number, line.line_number, line
        print("DEBUG: No relevant information found in any sentences")
        return "No relevant information found in the document.", None, None, None