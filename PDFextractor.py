import fitz
import PIL.Image
from io import BytesIO

class PDFextractor:
    def __init__(self):
        pass

    def get_text_queue(self, pdf_path):
        pdf = fitz.open(pdf_path)
        
        text_result = []
        for i in range(len(pdf)):
            page = pdf[i]
            texts = page.get_text("blocks")
            text_result.extend([text[4] for text in texts])

        image_result = []
        for i in range(len(pdf)):
            page = pdf[i]
            images = page.get_images()
            image_result.extend([PIL.Image.open(BytesIO(pdf.extract_image(image[0])["image"])) for image in images])

        return (text_result, image_result)