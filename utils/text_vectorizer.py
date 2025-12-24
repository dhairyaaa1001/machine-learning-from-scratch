def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c.isalpha() or c == ' '])
    return text

def build_vocab(texts):
    return list(set(" ".join(texts).split()))

def text_to_features(text, vocab):
    words = text.split()
    return [words.count(word) for word in vocab]
