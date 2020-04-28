import numpy as np


def adversarial_text(raw, nlp, indices, transform):
    adv_words = [token.text_with_ws for token in nlp(raw)]
    for i in indices:
        if i >= len(adv_words): continue
        adv_words[i] = transform(adv_words[i])
    return ''.join(adv_words)


homos = {'-':'˗','9':'৭','8':'Ȣ','7':'𝟕','6':'б','5':'Ƽ','4':'Ꮞ','3':'Ʒ','2':'ᒿ','1':'l','0':'O',
         "'":'`','a': 'ɑ', 'b': 'Ь', 'c': 'ϲ', 'd': 'ԁ', 'e': 'е', 'f': '𝚏', 'g': 'ɡ', 'h': 'հ',
         'i': 'і', 'j': 'ϳ', 'k': '𝒌', 'l': 'ⅼ', 'm': 'ｍ', 'n': 'ո', 'o':'о', 'p': 'р', 'q': 'ԛ',
         'r': 'ⲅ', 's': 'ѕ', 't': '𝚝', 'u': 'ս', 'v': 'ѵ', 'w': 'ԝ', 'x': '×', 'y': 'у', 'z': 'ᴢ'}


def homoglyph(word):
    N = len(word)-1 if word[-1] == ' ' else len(word)
    N = max(1, N)
    s = np.random.randint(0, N)
    if word[s] in homos: 
        adv_char = homos[word[s]]
    else:
        adv_char = word[s]
    adv_word = word[:s] + adv_char + word[s+1:]
    return adv_word


def remove_char(word):
    N = len(word)-1 if word[-1] == ' ' else len(word)
    N = max(1, N)
    s = np.random.randint(0, N)
    adv_word = word[:s] + word[s+1:]
    return adv_word


def flip_char(word):
    N = len(word)-1 if word[-1] == ' ' else len(word)
    N = max(1, N)
    s = np.random.randint(0, N)
    letter = ord(word[s])
    adv_char = np.random.randint(0,25) + 97
    adv_word = word[:s] + chr(adv_char) + word[s+1:]
    return adv_word