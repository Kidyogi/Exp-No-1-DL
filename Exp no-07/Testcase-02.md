code :

normal_translation_dict = {
    "He": "वह",
    "is": "है",
    "reading": "पढ़",
    "a": "एक",
    "book": "किताब"
}

attention_translation_dict = {
    "He": "वह",
    "is": "है",
    "reading": "पढ़",
    "a": "एक",
    "book": "पुस्तक"  
}

def normal_translate(sentence):
    words = sentence.split()
    translated = [normal_translation_dict.get(word, word) for word in words]
    return " ".join(translated) + "।"

def attention_translate(sentence):
    words = sentence.split()
    translated = [attention_translation_dict.get(word, word) for word in words]
    return " ".join(translated) + "।"

input_sentence = "He is reading a book"
normal_output = normal_translate(input_sentence)
attention_output = attention_translate(input_sentence)
print("Input Sentence: ", input_sentence)
print("Normal Output (Without Attention): ", normal_output)
print("Attention Output (With Attention): ", attention_output)

output :

<img width="357" height="68" alt="Screenshot 2025-10-15 114404" src="https://github.com/user-attachments/assets/62124ed2-044e-416d-b566-3c6a7f3fb2ce" />
