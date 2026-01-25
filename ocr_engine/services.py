import cv2
from PIL import Image
import pytesseract
from documents.models import OCRLine

# Configure pytesseract (assuming tesseract is installed and in PATH)
# For Windows, might need: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    """
    Preprocess the image using OpenCV for better OCR.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Save preprocessed image
    preprocessed_path = image_path.replace('.png', '_processed.png')
    cv2.imwrite(preprocessed_path, thresh)
    return preprocessed_path

def perform_ocr(image_path):
    """
    Perform OCR on the image using Tesseract (CRNN-based).
    Returns a list of line texts.
    """
    # Preprocess
    processed_path = preprocess_image(image_path)
    # Load image
    image = Image.open(processed_path)
    # Get data
    data = pytesseract.image_to_data(image, lang='eng', output_type=pytesseract.Output.DICT)
    # Group by lines
    lines = []
    current_line_num = -1
    current_line_texts = []
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if text:
            line_num = data['line_num'][i]
            if line_num != current_line_num:
                if current_line_texts:
                    lines.append(' '.join(current_line_texts))
                current_line_texts = [text]
                current_line_num = line_num
            else:
                current_line_texts.append(text)
    if current_line_texts:
        lines.append(' '.join(current_line_texts))
    return lines

def save_ocr_lines(page, lines):
    """
    Save the OCR lines to the database.
    """
    for i, line_text in enumerate(lines, start=1):
        if line_text.strip():  # Skip empty lines
            OCRLine.objects.create(
                page=page,
                line_number=i,
                text=line_text.strip()
            )