from pyvi.ViTokenizer import ViTokenizer

def vi_tokenize(text):
    tokens = ViTokenizer.tokenize(text)
    token_list = tokens.split()
    return token_list