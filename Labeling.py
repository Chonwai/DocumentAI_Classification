import numpy as np
import glob
import pickle

from sentence_transformers import SentenceTransformer

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, entropy_sampling

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB, ComplementNB, BernoulliNB

from IPython import display
from matplotlib import pyplot as plt

embedder = SentenceTransformer('distiluse-base-multilingual-cased-v2')
X_train_corpus = []
X_test_corpus = []
corpus = []
y_train = []
y_test = [1, 2, 1, 2, 3, 3]
modelName = 'model.pkl'

def random_sampling(classifier, X_pool):
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples))
    return query_idx, X_pool[query_idx]

for filename in glob.glob('./train/*.txt'):
    display.clear_output(wait=True)
    f = open(filename, "r")
    documentText = f.read().replace('\n', '').replace(' ', '')
    print(documentText)
    input_value = np.array([int(input())], dtype=int)
    y_train.append(input_value[0])
    X_train_corpus.append(documentText)


X_train = embedder.encode(X_train_corpus)


for filename in glob.glob('./pool/*.txt'):
    f = open(filename, "r")
    documentText = f.read().replace('\n', '').replace(' ', '')
    corpus.append(documentText)

corpus_embeddings = embedder.encode(corpus)


for filename in glob.glob('./test/*.txt'):
    f = open(filename, "r")
    documentText = f.read().replace('\n', '').replace(' ', '')
    X_test_corpus.append(documentText)

X_test = embedder.encode(X_test_corpus)

print(X_train.shape, y_train)
print(X_test.shape)

learner = ActiveLearner(
    estimator=RandomForestClassifier(n_jobs=4),
    query_strategy=entropy_sampling,
    X_training=X_train, y_training=y_train
)

n_queries = 5

accuracy_scores = []

for i in range(n_queries):
    display.clear_output(wait=True)
    query_idx, query_inst = learner.query(corpus_embeddings)
    print(corpus[query_idx[0]])
    print('\n')
    print("AI guess it is category: {category}".format(
        category=learner.predict(query_inst)[0]))
    print("Which category is this?")
    y_new = np.array([int(input())], dtype=int)
    learner.teach(query_inst.reshape(1, -1), y_new)
    score = learner.score(X_test, y_test)
    accuracy_scores.append(score)

with open(modelName, 'wb') as file:
    pickle.dump(learner, file)

print(accuracy_scores)