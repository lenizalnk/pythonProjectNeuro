import pickle
from pymorphy2 import MorphAnalyzer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from main import normalize_text
import nltk

# подгружаем необходимые словари
nltk.download('punkt')
nltk.download('stopwords')
morph = MorphAnalyzer()

# загружаем сохраненные модель и векторизатор
model = pickle.load(open("logref_model.sav", 'rb'))
vectorizer: TfidfVectorizer = pickle.load(open("vectorizer.pk", "rb"))


def get_intent(text):
    # векторизируем текст

    vector = vectorizer.transform([normalize_text(text)]).toarray()

    probability_distribution = model.predict_proba(vector)
    print(probability_distribution)

    if np.max(probability_distribution) > 0.5:
        return model.predict(vector)[0]


if __name__ == '__main__':
    print(get_intent("вы делаете обивку кожей?"))