import easyocr
import re

reader = easyocr.Reader(['en'])

def read_plate(img):
    result = reader.readtext(
        img,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )

    if result:
        text = result[0][1]
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        return text

    return "UNKNOWN"