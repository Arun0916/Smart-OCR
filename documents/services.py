import os
import fitz  # PyMuPDF
from PIL import Image
from django.core.files.base import ContentFile
from django.conf import settings
from documents.models import Document, Page

def process_uploaded_file(document):
    """
    Process the uploaded file: if PDF, convert pages to images and create Page objects.
    If image, create a single Page object.
    """
    file_path = document.uploaded_file.path
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        # Open PDF
        pdf_document = fitz.open(file_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.open(ContentFile(pix.tobytes()))
            # Save image
            image_name = f"{document.title}_page_{page_num + 1}.png"
            image_path = f"pages/{image_name}"
            full_image_path = os.path.join(settings.MEDIA_ROOT, image_path)
            os.makedirs(os.path.dirname(full_image_path), exist_ok=True)
            img.save(full_image_path)
            # Create Page object
            Page.objects.create(
                document=document,
                page_number=page_num + 1,
                image=image_path
            )
        pdf_document.close()
    elif file_extension in ['.jpg', '.jpeg', '.png']:
        # For images, create one page
        img = Image.open(file_path)
        image_name = f"{document.title}_page_1.png"
        image_path = f"pages/{image_name}"
        full_image_path = os.path.join(settings.MEDIA_ROOT, image_path)
        os.makedirs(os.path.dirname(full_image_path), exist_ok=True)
        img.save(full_image_path)
        Page.objects.create(
            document=document,
            page_number=1,
            image=image_path
        )
    else:
        raise ValueError("Unsupported file type")