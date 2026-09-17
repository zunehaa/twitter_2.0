import re

def clean_text(text):
    """
    Cleans the input text for sentiment analysis.
    This function is identical to the one used during model training
    to ensure 100% compatibility with the trained model.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # URLs
    text = re.sub(r"@\w+", " ", text)                     # mentions
    text = re.sub(r"#(\w+)", r"\1", text)                 # hashtags -> keep word
    text = re.sub(r"[^a-z\s']", " ", text)                 # keep letters/apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    return text
