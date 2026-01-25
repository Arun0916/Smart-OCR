# OCR + Q&A Django System

A Django-based system for uploading documents (images/PDFs), performing OCR on Tamil and English text, and answering questions based on the extracted text.

## Features

- Upload images or PDFs
- Automatic PDF to image conversion
- Model-based OCR using Tesseract (CRNN/LSTM-based)
- Store OCR text page-wise and line-wise
- Question-Answering using embeddings and FAISS
- Django Admin for management
- Simple frontend with templates

## Setup

1. **Python Version**: Use Python 3.8-3.12 (PyTorch does not support 3.13 yet). If using 3.13, consider downgrading.

2. Install Tesseract OCR:
   - Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Install and add to PATH
   - For Tamil support, download tam.traineddata and place in tessdata folder

3. Install dependencies:
   - Packages: Django, Pillow, opencv-python, pytesseract, faiss-cpu, sentence-transformers, pymupdf, numpy

4. Run migrations:
   ```
   python manage.py makemigrations
   python manage.py migrate
   ```

5. Create superuser:
   ```
   python manage.py createsuperuser
   ```

6. Run server:
   ```
   python manage.py runserver
   ```

7. Access:
   - Admin: /admin/
   - Upload: /documents/upload/
   - List: /documents/list/
   - Ask Q: /qa/ask/

## Notes

- OCR uses Tesseract for CRNN-style recognition.
- Embeddings use multilingual Sentence Transformers.
- FAISS index rebuilt on each query (optimize for production by caching index).
- For better Tamil OCR, fine-tune Tesseract or use a custom model."# Smart-OCR" 
