import torch
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import spacy
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, n_layers=1, bidirectional=False, dropout=0.2):
        super().__init__()
        if bidirectional:
            self.bi = 2
        else:
            self.bi = 1
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, batch_first=True,
                            bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.Linear(hidden_size * self.bi, 1)
        self.fc = nn.Linear(hidden_size * self.bi, 10)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, (ht1, ct1) = self.lstm(x)
        attention_weights = torch.softmax(self.attention(out), dim=1)
        attended_vectors = attention_weights * out
        context_vector = torch.sum(attended_vectors, dim=1)
        output = self.fc(context_vector)
        return output


spacy_eng = spacy.load("en_core_web_sm")


class Vocabulary:
    def __init__(self):
        self.stoi = {}

    def __len__(self):
        return len(self.stoi)

    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def load_vocab(self):
        with open('review/stoi.txt') as stoi:
            for i in stoi.readlines():
                key, val = i.strip().split(':')
                self.stoi[key] = int(val)

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [1] + [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text] + [2]


nltk.download('stopwords')
nltk.download('wordnet')


def lemmatize_text(text):
    lemm = WordNetLemmatizer()
    text = text.split()
    text = list(map(lemm.lemmatize, text))
    return ' '.join(text)


def remove_stopwords(text):
    stop_words = stopwords.words("english")
    no_stop = []
    for word in text.split(' '):
        if word not in stop_words:
            no_stop.append(word)
    return " ".join(no_stop)


def remove_punctuation_func(text):
    return re.sub(r'[^a-zA-Z0-9]', ' ', text)


def text_preprocessing(text):
    text = text.lower()
    text = remove_stopwords(text)
    text = remove_punctuation_func(text)
    text = lemmatize_text(text)
    text = re.sub(r'\bbr\b', '', text)
    text = re.sub(r"\s+", " ", text)
    return text


def movie_review_page(request):
    return render(request, 'review.html')


vocab = Vocabulary()
vocab.load_vocab()

model = Model(len(vocab), 256, 256, 4, True, 0.4)
model.load_state_dict(torch.load('review/model_weights.pth', map_location=torch.device('cpu')))
model.eval()


@csrf_exempt
def predict_review(request):
    if request.method == 'POST':
        review_text = request.POST.get('review_text')

        preprocessed_review = text_preprocessing(review_text)
        preprocessed_review = torch.tensor(vocab.numericalize(preprocessed_review)).unsqueeze(0)

        rating = model(preprocessed_review).argmax(1).item()
        sentiment = "It's a positive review" if rating > 5 else "It's a negative review"

        response = {
            'rating': rating,
            'sentiment': sentiment
        }

        return JsonResponse(response)

    return JsonResponse({'error': 'Invalid request method.'})
