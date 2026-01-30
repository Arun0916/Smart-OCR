
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .models import Page,Document
from .utils import get_gemini_response

# generate_answer import safe
# try:
#     from .services import get_gemini_response
#     QA_AVAILABLE = True
# except Exception:
#     QA_AVAILABLE = False



def ask_question(request):
    if request.method == 'POST':
        question_text = request.POST.get('question', '').strip()
        doc_id = request.POST.get('document_id') # Form-la irundhu doc_id varanum

        if not question_text:
            return render(request, 'qa_engine/answer.html', {'answer': 'Please enter a question.'})

        # --- GEMINI CONNECTION START ---
        try:
            # 1. Database-la irundhu images-ai edukkurom
            # Inga neenga document_id-ai filter pannanum
            pages = Document.objects.filter(document_id=doc_id) if doc_id else Page.objects.all()

            # 2. Gemini function-ai call panrom
            answer = get_gemini_response(pages, question_text)
            
        except Exception as e:
            answer = f"Gemini Error: {str(e)}"
        # --- GEMINI CONNECTION END ---

        # Question save panra logic
        # Question.objects.create(...) 

        return render(request, 'qa_engine/answer.html', {
            'question': question_text,
            'answer': answer,
        })

    return render(request, 'qa_engine/ask.html')