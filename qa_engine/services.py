import faiss
import numpy as np
import re
from documents.models import OCRLine
import json
import os

# Try to import sentence_transformers, but allow graceful degradation if unavailable
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, OSError) as e:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    print(f"WARNING: SentenceTransformers not available: {e}")

# Cache for the model to avoid reloading
_MODEL_CACHE = None
# Cache for the FAISS index to avoid rebuilding
_INDEX_CACHE = None
_EMBEDDINGS_CACHE = None
_IDS_CACHE = None

# Check if LLM is enabled (API key available)
LLM_ENABLED = os.environ.get('ANTHROPIC_API_KEY') is not None

def _get_cached_model():
    """
    Get or create cached SentenceTransformer model.
    Internal helper function.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is None and SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            _MODEL_CACHE = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("DEBUG: SentenceTransformer model loaded successfully")
        except Exception as e:
            print(f"DEBUG: Failed to load SentenceTransformer model: {e}")
            return None
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

def get_all_texts(document_id=None):
    """
    Retrieve all OCR line texts and their IDs.
    If document_id is provided, only get lines from that document.
    """
    if document_id:
        lines = OCRLine.objects.select_related('page__document').filter(
            page__document_id=document_id
        ).all()
    else:
        lines = OCRLine.objects.select_related('page__document').all()
    
    texts = [line.text for line in lines]
    ids = [line.id for line in lines]
    print(f"DEBUG: Retrieved {len(texts)} OCR lines" + (f" for document {document_id}" if document_id else ""))
    return texts, ids

def build_index(document_id=None, force_rebuild=False):
    """
    Build FAISS index for all OCR texts.
    Returns (index, embeddings, ids) or (None, None, None) if unsuccessful.
    
    Args:
        document_id: Optional document ID to limit search scope
        force_rebuild: Force rebuilding the index even if cached
    """
    global _INDEX_CACHE, _EMBEDDINGS_CACHE, _IDS_CACHE
    
    # Check cache
    if not force_rebuild and _INDEX_CACHE is not None:
        print("DEBUG: Using cached FAISS index")
        return _INDEX_CACHE, _EMBEDDINGS_CACHE, _IDS_CACHE
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("DEBUG: SentenceTransformers not available")
        return None, None, None
    
    # Use cached model
    model = _get_cached_model()
    if model is None:
        print("DEBUG: Model not available")
        return None, None, None
    
    texts, ids = get_all_texts(document_id)
    if not texts:
        print("DEBUG: No texts found to index")
        return None, None, None
    
    print(f"DEBUG: Building FAISS index for {len(texts)} texts...")
    try:
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        dimension = embeddings.shape[1]
        print(f"DEBUG: Embeddings shape: {embeddings.shape}, dimension: {dimension}")
        
        index = faiss.IndexFlatIP(dimension)  # Cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        print(f"DEBUG: FAISS index built successfully with {index.ntotal} vectors")
        
        # Cache the index
        _INDEX_CACHE = index
        _EMBEDDINGS_CACHE = embeddings
        _IDS_CACHE = ids
        
        return index, embeddings, ids
    except Exception as e:
        print(f"DEBUG: Error building FAISS index: {e}")
        return None, None, None

def search_relevant_texts(question, document_id=None, top_k=10):
    """
    Search for the most relevant OCR lines to the question using semantic similarity.
    Returns list of (OCRLine, score)
    
    Args:
        question: The question to search for
        document_id: Optional document ID to limit search scope
        top_k: Number of top results to return
    """
    print(f"DEBUG: search_relevant_texts called with question='{question}', document_id={document_id}, top_k={top_k}")
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("DEBUG: SentenceTransformers not available, falling back to keyword search")
        return keyword_search_fallback(question, document_id, top_k)
    
    index, embeddings, ids = build_index(document_id)
    if index is None or embeddings is None or ids is None:
        print("DEBUG: FAISS index unavailable, falling back to keyword search")
        return keyword_search_fallback(question, document_id, top_k)
    
    # Use cached model
    model = _get_cached_model()
    if model is None:
        print("DEBUG: Model unavailable, falling back to keyword search")
        return keyword_search_fallback(question, document_id, top_k)
    
    try:
        query_embedding = model.encode([question], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(query_embedding)
        print(f"DEBUG: Query embedding shape: {query_embedding.shape}")
        
        distances, indices = index.search(query_embedding, min(top_k, index.ntotal))
        print(f"DEBUG: Search returned {len(distances[0])} results")
        print(f"DEBUG: Top 3 distances: {distances[0][:3]}")
        print(f"DEBUG: Top 3 indices: {indices[0][:3]}")
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(ids):
                line_id = ids[idx]
                try:
                    line = OCRLine.objects.get(id=line_id)
                    results.append((line, float(dist)))
                    print(f"DEBUG: Found line {line_id} with score {dist:.3f}: {line.text[:80]}")
                except OCRLine.DoesNotExist:
                    print(f"DEBUG: OCRLine {line_id} not found in database")
                    continue
        
        if not results:
            print("DEBUG: No results from semantic search, falling back to keyword search")
            return keyword_search_fallback(question, document_id, top_k)
        
        print(f"DEBUG: Returning {len(results)} results from semantic search")
        return results
    
    except Exception as e:
        print(f"DEBUG: Error in semantic search: {e}")
        return keyword_search_fallback(question, document_id, top_k)

def keyword_search_fallback(question, document_id=None, top_k=10):
    """
    Fallback keyword-based search when semantic search is unavailable.
    Returns list of (OCRLine, score)
    """
    print(f"DEBUG: Using keyword search fallback for question: '{question}'")
    
    # Get all lines
    if document_id:
        lines = OCRLine.objects.select_related('page__document').filter(
            page__document_id=document_id
        ).all()
    else:
        lines = OCRLine.objects.select_related('page__document').all()
    
    if not lines.exists():
        print("DEBUG: No OCR lines found in database")
        return []
    
    # Extract keywords from question
    question_lower = question.lower()
    stop_words = {'what', 'is', 'the', 'a', 'an', 'and', 'or', 'find', 'get', 'show', 'tell', 'me'}
    keywords = [word for word in question_lower.split() if word not in stop_words and len(word) > 2]
    
    # Add synonyms and OCR variations for common terms
    synonym_map = {
        'address': ['address', 'addr', 'addres', 'adress', 'postal', 'post', 'paste', 'location', 'place'],
        'account': ['account', 'acc', 'acct', 'a/c', 'ac', 'sb'],
        'email': ['email', 'emai', 'mail', 'e-mail', 'e mail'],
        'phone': ['phone', 'phon', 'mobile', 'cell', 'contact', 'number', 'naber', 'no'],
        'name': ['name', 'nam', 'applicant', 'customer'],
        'bank': ['bank', 'banc', 'branch'],
    }
    
    # Expand keywords with synonyms
    expanded_keywords = set(keywords)
    for keyword in keywords:
        if keyword in synonym_map:
            expanded_keywords.update(synonym_map[keyword])
        # Also add common OCR variations (missing first/last char)
        if len(keyword) > 4:
            expanded_keywords.add(keyword[1:])  # Missing first char
            expanded_keywords.add(keyword[:-1])  # Missing last char
    
    keywords = list(expanded_keywords)
    print(f"DEBUG: Expanded keywords: {keywords}")
    
    # Score each line
    results = []
    for line in lines:
        if len(line.text.strip()) < 3:
            continue
        
        text_lower = line.text.lower()
        score = 0
        matched_words = []
        
        # Check for keyword matches
        for keyword in keywords:
            if keyword in text_lower:
                score += 2
                matched_words.append(keyword)
            # Partial match (for OCR errors) - fuzzy matching
            elif len(keyword) > 3:
                # Check if keyword appears with 1-2 character differences
                for i in range(len(text_lower) - len(keyword) + 2):
                    if i + len(keyword) <= len(text_lower):
                        substr = text_lower[i:i+len(keyword)]
                        if len(substr) >= len(keyword) - 1:
                            # Calculate similarity
                            matches = sum(1 for a, b in zip(keyword, substr) if a == b)
                            if matches >= len(keyword) - 1:
                                score += 1
                                matched_words.append(f"{keyword}~{substr}")
                                break
        
        # Bonus: if line contains numbers/special patterns relevant to question type
        if any(kw in keywords for kw in ['address', 'addr', 'postal', 'paste']):
            # Address lines often have numbers and place names
            if re.search(r'\d+', text_lower) and any(place in text_lower for place in ['nagar', 'street', 'road', 'colony', 'chana', 'chennai', 'delhi']):
                score += 1
        
        if score > 0:
            results.append((line, float(score)))
            if matched_words:
                print(f"DEBUG: Line matched with keywords {matched_words}: {line.text[:60]}")
    
    # Sort by score
    results.sort(key=lambda x: -x[1])
    print(f"DEBUG: Keyword search found {len(results)} results, returning top {min(top_k, len(results))}")
    
    if results:
        for i, (line, score) in enumerate(results[:3]):
            print(f"DEBUG: Result {i+1} (score={score:.1f}): {line.text[:80]}")
    
    return results[:top_k]

def simple_rule_based_extraction(question, context_lines):
    """
    Simple rule-based extraction as fallback when LLM is not available.
    
    Args:
        question: User's question
        context_lines: List of (OCRLine, score) tuples
    
    Returns:
        dict with 'answer', 'confidence', 'source_line_number', 'reasoning'
    """
    if not context_lines:
        return {
            "answer": None,
            "confidence": "low",
            "source_line_number": None,
            "reasoning": "No relevant context found"
        }
    
    question_lower = question.lower()
    
    # Check all context lines, not just the first one
    all_text = "\n".join([line.text for line, score in context_lines[:5]])
    
    # Get the best matching line for fallback
    best_line, best_score = context_lines[0]
    text = best_line.text
    text_lower = text.lower()
    
    # Simple extraction patterns for common questions
    answer = None
    confidence = "medium"
    reasoning = ""
    source_line = 1
    
    # Account number patterns
    if any(kw in question_lower for kw in ['account', 'acc', 'a/c']):
        # Search through all context lines for account numbers
        for i, (line, score) in enumerate(context_lines[:5]):
            line_text = line.text
            # Look for long number sequences (account numbers are typically 9-18 digits)
            numbers = re.findall(r'\d{9,18}', line_text.replace(' ', '').replace('-', ''))
            if numbers:
                answer = numbers[0]
                reasoning = f"Found account number pattern in: {line_text[:80]}"
                source_line = i + 1
                break
        
        # If not found, look for IFSC codes or other account identifiers
        if not answer:
            for i, (line, score) in enumerate(context_lines[:5]):
                line_text = line.text
                # IFSC pattern
                ifsc_match = re.search(r'[A-Z]{4}0[A-Z0-9]{6}', line_text.upper())
                if ifsc_match:
                    answer = ifsc_match.group(0)
                    reasoning = f"Found IFSC code in: {line_text[:80]}"
                    source_line = i + 1
                    break
    
    # Email patterns
    elif any(kw in question_lower for kw in ['email', 'mail', 'e-mail']):
        # Search through all context lines for email
        for i, (line, score) in enumerate(context_lines[:5]):
            line_text = line.text
            # Standard email pattern
            email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', line_text)
            if email_match:
                answer = email_match.group(0).lower()
                reasoning = f"Found email pattern in: {line_text[:80]}"
                source_line = i + 1
                break
            
            # Handle OCR errors: look for patterns like "example Com" -> "example.com"
            # Pattern: word + space + Com/Net/Org
            ocr_email = re.search(r'([a-z0-9]+(?:[._][a-z0-9]+)*)\s*@?\s*([a-z0-9]+(?:[._][a-z0-9]+)*)\s+(com|net|org|in|co)', line_text.lower())
            if ocr_email:
                # Reconstruct email
                answer = f"{ocr_email.group(1)}@{ocr_email.group(2)}.{ocr_email.group(3)}"
                reasoning = f"Found email with OCR errors, reconstructed from: {line_text[:80]}"
                source_line = i + 1
                confidence = "low"
                break
    
    # Phone/Mobile patterns
    elif any(kw in question_lower for kw in ['phone', 'mobile', 'number', 'contact']):
        # Search through all context lines for phone numbers
        for i, (line, score) in enumerate(context_lines[:5]):
            line_text = line.text
            # Look for 10-digit phone numbers (Indian format)
            phone_numbers = re.findall(r'[6-9]\d{9}', line_text.replace(' ', '').replace('-', ''))
            if phone_numbers:
                answer = phone_numbers[0]
                reasoning = f"Found phone number pattern in: {line_text[:80]}"
                source_line = i + 1
                break
    
    # Name patterns
    elif any(kw in question_lower for kw in ['name', 'applicant', 'customer']):
        # Search through all context lines for names
        for i, (line, score) in enumerate(context_lines[:5]):
            line_text = line.text
            
            # Pattern 1: "Name: Arun Kumar" or "Fa Name "Arun Kumar"
            name_patterns = [
                r'(?:name|applicant|customer)[^"]*[":]\s*"?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'(?:Fa|Father)\s+Name[^"]*"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'Name\s+"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            ]
            
            for pattern in name_patterns:
                match = re.search(pattern, line_text)
                if match:
                    answer = match.group(1).strip()
                    reasoning = f"Found name pattern in: {line_text[:80]}"
                    source_line = i + 1
                    break
            
            if answer:
                break
        
        # Fallback: Look for quoted capitalized names
        if not answer:
            for i, (line, score) in enumerate(context_lines[:5]):
                line_text = line.text
                # Look for quoted names like "Arun Kumar"
                quoted_name = re.search(r'"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})"', line_text)
                if quoted_name:
                    answer = quoted_name.group(1)
                    reasoning = f"Found quoted name in: {line_text[:80]}"
                    source_line = i + 1
                    break
    
    # Bank name patterns
    elif 'bank' in question_lower:
        # Search through all context lines for bank name
        for i, (line, score) in enumerate(context_lines[:5]):
            line_text = line.text
            
            # Pattern 1: "Bank Name: ..." or "Bank Name 'State Bank..."
            match = re.search(r"bank\s+name[^:'\"]*[:'\"]\s*([^\n]+)", line_text.lower(), re.IGNORECASE)
            if match:
                answer = match.group(1).strip().strip("'\"")
                # Clean up - take reasonable words
                words = answer.split()[:6]
                answer = ' '.join(words)
                reasoning = f"Extracted bank name from: {line_text[:80]}"
                source_line = i + 1
                break
            
            # Pattern 2: Look for common bank names
            bank_patterns = [
                r"(state bank[^,.\n\"']{0,30})",
                r"(hdfc\s+bank[^,.\n\"']{0,20})",
                r"(icici\s+bank[^,.\n\"']{0,20})",
                r"(axis\s+bank[^,.\n\"']{0,20})",
            ]
            
            for pattern in bank_patterns:
                match = re.search(pattern, line_text, re.IGNORECASE)
                if match:
                    answer = match.group(1).strip()
                    words = answer.split()[:6]
                    answer = ' '.join(words)
                    reasoning = f"Found bank name pattern in: {line_text[:80]}"
                    source_line = i + 1
                    break
            
            if answer:
                break
    
    # Address patterns
    elif any(kw in question_lower for kw in ['address', 'addr', 'location']):
        # Search through all context lines for address
        for i, (line, score) in enumerate(context_lines[:5]):
            line_text = line.text
            
            # Pattern 1: "Address:" or "Postal:" or OCR variations like "Paste"
            address_patterns = [
                r"(?:address|addr|postal|post|paste)[^:]*[:]\s*(.+)",
                r"(?:ro|no)[.\s]+(\d+[^,]+(?:nagar|street|road|colony|avenue|lane)[^,\n]{0,100})",
            ]
            
            for pattern in address_patterns:
                match = re.search(pattern, line_text, re.IGNORECASE)
                if match:
                    answer = match.group(1).strip()
                    # Clean up - take reasonable portion
                    if len(answer) > 150:
                        answer = answer[:150]
                    # Remove trailing punctuation
                    answer = answer.rstrip('.,;')
                    reasoning = f"Extracted address from: {line_text[:80]}"
                    source_line = i + 1
                    break
            
            if answer:
                break
        
        # Fallback: Look for lines with address indicators (numbers + place names)
        if not answer:
            for i, (line, score) in enumerate(context_lines[:5]):
                line_text = line.text
                # Look for patterns like "45, Anna Nagar" or "Ro. 45, Anna Nagar"
                if re.search(r'\d+', line_text) and any(place in line_text.lower() for place in ['nagar', 'street', 'road', 'colony', 'chana', 'chennai', 'anna']):
                    # Extract the relevant portion
                    answer = line_text.strip()
                    if len(answer) > 150:
                        answer = answer[:150]
                    reasoning = f"Found address-like content in: {line_text[:80]}"
                    source_line = i + 1
                    confidence = "low"
                    break
    
    # Generic fallback: return the relevant line text
    if not answer:
        # Return first 100 chars of the most relevant line
        answer = text[:100].strip()
        confidence = "low"
        reasoning = f"No specific pattern found, returning relevant text: {text[:80]}"
    
    return {
        "answer": answer,
        "confidence": confidence,
        "source_line_number": str(source_line),
        "reasoning": reasoning
    }

def call_llm_for_extraction(question, context_lines):
    """
    Call Claude API to extract answer from OCR text context.
    Falls back to simple rule-based extraction if LLM is not available.
    
    Args:
        question: User's question
        context_lines: List of (OCRLine, score) tuples with relevant text
    
    Returns:
        dict with 'answer', 'confidence', 'source_line_id', 'reasoning'
    """
    if not context_lines:
        return {
            "answer": None,
            "confidence": "low",
            "source_line_number": None,
            "reasoning": "No context lines provided"
        }
    
    # Check if LLM is enabled
    if not LLM_ENABLED:
        print("DEBUG: LLM not enabled or API key not configured, using rule-based extraction")
        print(f"DEBUG: Using rule-based extraction for question: '{question}'")
        result = simple_rule_based_extraction(question, context_lines)
        print(f"DEBUG: Extraction result:")
        print(f"  Answer: {result.get('answer')}")
        print(f"  Confidence: {result.get('confidence')}")
        print(f"  Reasoning: {result.get('reasoning')}")
        return result
    
    # Build context from relevant lines
    context_text = "\n".join([
        f"[Line {i+1}] {line.text}" 
        for i, (line, score) in enumerate(context_lines[:10])  # Limit to top 10 for context window
    ])
    
    # Create prompt for Claude
    prompt = f"""You are analyzing OCR-extracted text from a document to answer a specific question.

OCR Text Context:
{context_text}

Question: {question}

Instructions:
1. Carefully read the OCR text which may contain errors (misspellings, spacing issues, etc.)
2. Find the most relevant information that answers the question
3. Extract the precise answer, correcting any obvious OCR errors
4. If multiple pieces of information are relevant, extract the most complete/accurate one
5. Be concise - extract only the specific answer, not surrounding context

Please respond in JSON format:
{{
    "answer": "the extracted answer (or null if not found)",
    "confidence": "high/medium/low",
    "source_line_number": "which line number(s) from the context contain the answer",
    "reasoning": "brief explanation of how you found the answer and any OCR corrections made"
}}

If the answer cannot be found in the provided text, set answer to null and explain why in reasoning."""

    try:
        # Make API call to Claude
        import requests
        
        print("DEBUG: Making API call to Claude...")
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=30
        )
        
        print(f"DEBUG: API response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            # Extract text content from response
            content = result.get('content', [])
            text_content = ''.join([
                item['text'] for item in content if item.get('type') == 'text'
            ])
            
            print(f"DEBUG: API response content: {text_content[:200]}")
            
            # Parse JSON response
            # Remove markdown code blocks if present
            text_content = text_content.strip()
            if text_content.startswith('```'):
                # Remove opening ```json or ```
                text_content = re.sub(r'^```(?:json)?\s*\n', '', text_content)
                # Remove closing ```
                text_content = re.sub(r'\n```\s*$', '', text_content)
            
            try:
                parsed_result = json.loads(text_content)
                print(f"DEBUG: Parsed LLM result: {parsed_result}")
                return parsed_result
            except json.JSONDecodeError as e:
                print(f"DEBUG: Failed to parse LLM response as JSON: {e}")
                print(f"DEBUG: Raw response: {text_content}")
                # Fallback to rule-based
                return simple_rule_based_extraction(question, context_lines)
        else:
            print(f"DEBUG: API call failed with status {response.status_code}: {response.text}")
            # Fallback to rule-based
            return simple_rule_based_extraction(question, context_lines)
    
    except Exception as e:
        print(f"DEBUG: Exception in LLM call: {str(e)}")
        import traceback
        traceback.print_exc()
        # Fallback to rule-based
        return simple_rule_based_extraction(question, context_lines)

def map_source_line_to_ocr_line(source_line_info, context_lines):
    """
    Map the source line number from LLM response back to actual OCRLine object.
    
    Args:
        source_line_info: String like "1", "2-3", or "Line 1"
        context_lines: List of (OCRLine, score) tuples
    
    Returns:
        OCRLine object or None
    """
    if not source_line_info or not context_lines:
        return None
    
    try:
        # Extract first number from the source line info
        numbers = re.findall(r'\d+', str(source_line_info))
        if numbers:
            line_idx = int(numbers[0]) - 1  # Convert to 0-indexed
            if 0 <= line_idx < len(context_lines):
                return context_lines[line_idx][0]
    except:
        pass
    
    # Fallback: return first line
    return context_lines[0][0] if context_lines else None

def generate_answer(question, document_id=None):
    """
    Generate answer from document OCR text using LLM-based extraction (with rule-based fallback).
    
    This function:
    1. Uses semantic search to find relevant OCR lines (with keyword fallback)
    2. Calls Claude API to extract precise answer from context (with rule-based fallback)
    3. Returns the answer with metadata
    
    Args:
        question: The question to answer
        document_id: Optional document ID to limit search scope
    
    Returns:
        tuple: (answer_text, page_number, line_number, line_obj)
    """
    print(f"DEBUG: generate_answer called with question: '{question}', document_id: {document_id}")
    
    # Step 1: Find relevant lines using semantic search (with fallback)
    results = search_relevant_texts(question, document_id, top_k=10)
    
    if not results:
        print("DEBUG: No relevant lines found even after fallback")
        return "No relevant information found in the document.", None, None, None
    
    print(f"DEBUG: Found {len(results)} relevant lines")
    for i, (line, score) in enumerate(results[:5]):
        print(f"DEBUG: Line {i+1} (score: {score:.3f}): {line.text[:100]}")
    
    # Step 2: Use LLM (or rule-based fallback) to extract answer from relevant context
    llm_result = call_llm_for_extraction(question, results)
    
    print(f"DEBUG: LLM extraction result:")
    print(f"  Answer: {llm_result.get('answer')}")
    print(f"  Confidence: {llm_result.get('confidence')}")
    print(f"  Reasoning: {llm_result.get('reasoning')}")
    
    # Step 3: Process LLM response
    answer = llm_result.get('answer')
    
    if answer and answer.lower() != 'null' and answer.strip():
        # Map back to source OCR line
        source_line = map_source_line_to_ocr_line(
            llm_result.get('source_line_number'),
            results
        )
        
        if source_line:
            return (
                answer,
                source_line.page.page_number,
                source_line.line_number,
                source_line
            )
        else:
            # Use first relevant line as fallback
            first_line = results[0][0]
            return (
                answer,
                first_line.page.page_number,
                first_line.line_number,
                first_line
            )
    else:
        # No answer found
        reasoning = llm_result.get('reasoning', 'Could not extract answer')
        
        # Try to provide helpful context from most relevant line
        if results:
            best_line = results[0][0]
            return (
                f"Could not find a clear answer. {reasoning}",
                best_line.page.page_number,
                best_line.line_number,
                best_line
            )
        else:
            return (
                "No relevant information found in the document.",
                None,
                None,
                None
            )

# Backward compatibility: keep old function names but delegate to new implementation
def find_relevant_lines(question, top_k=5, document_id=None):
    """
    Legacy function - now uses semantic search (with keyword fallback) instead of keyword matching.
    Returns list of (full_line_text, OCRLine, score) sorted by score desc.
    """
    results = search_relevant_texts(question, document_id, top_k)
    return [(line.text, line, score) for line, score in results]

def extract_answer(question, text):
    """
    Legacy function - now uses LLM extraction (with rule-based fallback).
    For compatibility, this creates a minimal context and calls the LLM.
    """
    # Create a mock OCRLine-like object for the context
    class MockLine:
        def __init__(self, text):
            self.text = text
    
    mock_line = MockLine(text)
    context = [(mock_line, 1.0)]
    
    llm_result = call_llm_for_extraction(question, context)
    return llm_result.get('answer')

def classify_question(question):
    """
    Legacy function - kept for backward compatibility but not used in LLM approach.
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