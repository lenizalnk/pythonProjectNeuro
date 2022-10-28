import pickle

from pymorphy2 import MorphAnalyzer
import pandas as pd
import nltk
from nltk.corpus import stopwords
import numpy as np

# загрузка модулей nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# nltk.download()
nltk.download('punkt')
nltk.download('stopwords')
# морфологический анализатор для русского языка
morph = MorphAnalyzer()

# считываем датасет
df = pd.read_csv("dataset.csv")
# print(df.text)

# подгружаем стоп слова для русского языка
stopwords_ru = stopwords.words("russian")

def normalize_text(text, stop_words=False):
    # приведение к нижнему регистру и токенизация
    text = nltk.word_tokenize(text.lower())

    # удаление стоп слов
    if stop_words:
        text = [token for token in text if token not in stopwords_ru]

    # лемматизация
    text = [morph.normal_forms(token)[0] for token in text]

    return " ".join(text)

print(normalize_text("Как работаете механизм топган", False))

# нормализируем текст

df["normal_text"] = [normalize_text(text, stop_words=True) for text in df.text]

vectorizer = TfidfVectorizer()

# text = vectorizer.fit_transform(df.normal_text)
# print(text)

# считаем векторное сходство фраз

from sklearn.feature_extraction.text import TfidfVectorizer

"""
TfidfVectorizer() работает следующим образов:
1. преобразует запросы с помощью CountVectorizer() - который суммирует one-hot эмбеддинги всех слов запроса
2. трансформирует полученные эмбеддинги, применяя tf*idf
"""
text_embeddings = vectorizer.fit_transform(df.text)
# print(text_embeddings)

# кластеризируем данные

# KMeans

# MiniBatchKMeans

from sklearn.cluster import KMeans, MiniBatchKMeans


def cluster_kmeans(num_clusters, embeddings, init='k-means++', random_state=42):
    clustering_model = KMeans(n_clusters=num_clusters, init=init, n_init=100, random_state=random_state)
    clusters = clustering_model.fit_predict(embeddings)
    return clusters


def cluster_miniBatchKMeans(num_clusters, embeddings, init_size=16, batch_size=16, random_state=42):
    clustering_model = MiniBatchKMeans(n_clusters=num_clusters, init_size=init_size, batch_size=batch_size,
                                       random_state=random_state)
    clusters = clustering_model.fit_predict(embeddings)
    return clusters


num_clusters = 5

kmeans = cluster_kmeans(num_clusters, text_embeddings)
miniBatchKMeans = cluster_miniBatchKMeans(num_clusters, text_embeddings)

# print(kmeans)
# print(miniBatchKMeans)

# визуализация

# импорт библиотек для визулизации кластеров
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_tsne_pca(embeddings, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(embeddings.shape[0]), size=200, replace=False)

    pca = PCA(n_components=2).fit_transform(embeddings[max_items, :].todense())
    tsne = TSNE(perplexity=15).fit_transform(PCA(n_components=20).fit_transform(embeddings[max_items, :].todense()))

    idx = np.random.choice(range(pca.shape[0]), size=200, replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i / max_label) for i in label_subset[idx]]

    f, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title('PCA')

    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title('TSNE')

    # f.show()
    plt.show()


plot_tsne_pca(text_embeddings, kmeans)
plot_tsne_pca(text_embeddings, miniBatchKMeans)

# находим самые частые слова в каждом кластере

def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    top_keywords = []
    for i, r in df.iterrows():
        print('\nCluster {}'.format(i))
        l = [labels[t] for t in np.argsort(r)[-n_terms:]]
        l.sort()
        print(','.join(l))
        top_keywords.append(','.join(l))

    return top_keywords


top_words = 1
cluster_names = get_top_keywords(text_embeddings, kmeans, vectorizer.get_feature_names(), top_words)
print(cluster_names)

# собираем кластеры в удобный нам вид

clustered_sentences = [[] for i in range(num_clusters)]

df["label"] = ["" for _ in range(len(df.text))]

for sentence_id, cluster_id in enumerate(kmeans):
    clustered_sentences[cluster_id].append(df.text[sentence_id])
    df.label[sentence_id] = cluster_names[cluster_id]

print(df.head())

# выводим их

for i in range(len(clustered_sentences)):
    print(cluster_names[i])
    print(clustered_sentences[i])

df.to_csv("data_marketed.csv")


from sklearn.model_selection import train_test_split

X = vectorizer.fit_transform(df.normal_text)
y = list(df.label)

# Разделение датасета на train и test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# =======
# KNeighborsClassifier
# =======

# Выбор оптимального числа соседей
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def find_optimal_k(X_train, y_train, max_k):
    iters = range(1, max_k, 1)
    acc = []
    for k in iters:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        acc.append(knn.score(X_test, y_test))

    f, ax = plt.subplots(1, 1)
    ax.plot(iters, acc, marker='o')
    ax.set_xlabel('number of K')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by K')

    # f.show()
    plt.show()

max_k = 10
find_optimal_k(X_train, y_train, max_k)

n_neighbors = 8
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))

y_true, y_pred = y_test, knn.predict(X_test)
print(classification_report(y_true, y_pred, zero_division=1))

## -- пример запроса к модели
result_vector = vectorizer.transform(['паралоном обиваете'])
print('Запрос: ', knn.predict_proba(result_vector))

# =======
# end KNeighborsClassifier
# =======

# =======
# LogisticRegressionCV
# =======

from sklearn.linear_model import LogisticRegressionCV

logreg_clf = LogisticRegressionCV(multi_class="multinomial")
logreg_clf.fit(X_train, y_train)

print(logreg_clf.score(X_test, y_test))

y_pred = logreg_clf.predict(X_test)
print(classification_report(y_true, y_pred, zero_division=1))

## -- пример запроса к модели
result_vector = vectorizer.transform(['паралоном обиваете'])
print('Запрос: ', logreg_clf.predict_proba(result_vector))

## -- сохранение рзультата в файл

model_filename = "logref_model.sav"
pickle.dump(logreg_clf, open(model_filename, 'wb'))
pickle.dump(vectorizer, open("vectorizer.pk", 'wb'))

print('Model is saved into to disk successfully Using Pickle')