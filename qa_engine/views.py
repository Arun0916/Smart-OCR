
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .models import Question

# generate_answer import safe
try:
    from .services import generate_answer
    QA_AVAILABLE = True
except Exception:
    QA_AVAILABLE = False



def ask_question(request):
    if request.method == 'POST':
        question_text = request.POST.get('question', '').strip()
        print(f"DEBUG: Question received in view: '{question_text}'")

        # Empty question check
        if not question_text:
            return render(request, 'qa_engine/answer.html', {
                'question': '',
                'answer': 'Please enter a valid question.',
                'page': None,
                'line': None
            })

        answer = "No relevant information found."
        page_num = None
        line_num = None
        line_obj = None

        if QA_AVAILABLE:
            try:
                result = generate_answer(question_text)

                # Support both tuple & string return
                if isinstance(result, tuple):
                    answer, page_num, line_num, line_obj = result
                else:
                    answer = result

            except (ImportError, OSError, RuntimeError) as e:
                answer = (
                    "Question-Answering system is temporarily unavailable "
                    "due to ML library issues. OCR data is still stored."
                )

        else:
            answer = (
                "ML Question-Answering module is not loaded. "
                "Please check Python & PyTorch compatibility."
            )

        # Save question safely
        Question.objects.create(
            user=request.user,
            question_text=question_text,
            answer_text=answer,
            page_reference=line_obj.page if line_obj else None,
            line_reference=line_obj if line_obj else None
        )

        return render(request, 'qa_engine/answer.html', {
            'question': question_text,
            'answer': answer,
            'page': page_num,
            'line': line_num
        })

    return render(request, 'qa_engine/ask.html')
