import re
import string

def preprocess_text(text):
    text = text.lower()

    text = re.sub(r'<.*?>', '', text)

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = re.sub(r'\d+', '', text)

    return text
