from google import genai
from PIL import Image

def get_gemini_response(page_objects, user_question):
    # Puthu SDK style
    client = genai.Client(api_key="AIzaSyCnsTov_4UwiKdv8_C_pPhwj_awqMU7uyU")
    
    content = [f"Please answer based on the images: {user_question}"]
    
    for page in page_objects:
        img = Image.open(page.image.path)
        content.append(img)

    try:
        # Puthu method 'generate_content' use pannunga
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=content
        )
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"