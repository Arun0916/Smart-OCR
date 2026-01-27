from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login, logout
from django.contrib import messages
from .forms import DocumentForm
from .models import Document
from .services import process_uploaded_file
from ocr_engine.services import perform_ocr, save_ocr_lines

def home(request):
    return render(request,'documents/index.html')


@login_required
def upload_document(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            document = form.save(commit=False)
            document.user = request.user
            document.save()
            # Process the uploaded file
            process_uploaded_file(document)
            # Perform OCR on each page
            for page in document.page_set.all():
                lines = perform_ocr(page.image.path)
                save_ocr_lines(page, lines)
            return redirect('document_list')
    else:
        form = DocumentForm()
    return render(request, 'documents/upload.html', {'form': form})

@login_required
def document_list(request):
    documents = Document.objects.filter(user=request.user)
    return render(request, 'documents/dashboard.html', {'documents': documents})

def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Account created successfully!')
            return redirect('upload_document')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = UserCreationForm()
    return render(request, 'registration/signup.html', {'form': form})

def logout_view(request):
    logout(request)
    messages.info(request, 'You have been logged out.')
    return redirect('login')