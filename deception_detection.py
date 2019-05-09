import numpy as np
import pandas as pd
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Dense, Conv1D, MaxPooling1D, Flatten, Embedding, GlobalMaxPool1D
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


def concat_datasets(train, test, validation):
    COLUMNS_LABELS = ['ID', 'Label', 'Statement', 'Subject', 'Speaker', "Speaker's job",
                     'State info', 'Party affiliation', 'Barely true counts', 'False counts',
                     'Half true counts', 'Mostly true counts', 'Pants on fire counts', 'Venue']
    
    train.columns = test.columns = validation.columns = COLUMNS_LABELS
    
    return pd.concat([train, test, validation])


def simplify_labels(label):
    true_labels = ['half-true', 'mostly-true', 'true']
    
    return 'true' if label in true_labels else 'false'


def split_dataframe(df):
    df['Label'] = df['Label'].apply(simplify_labels)
    
    X = df.iloc[:, 2].values
    y = df.iloc[:, 1].values
    
    return X, y


def stemming_documents(documents):
    whitespace_tokenizer = WhitespaceTokenizer()
    stemmer = PorterStemmer()
    stemmed_documents = []
    
    for document in documents:
        sentence = ' '.join([stemmer.stem(word.lower()) for word in whitespace_tokenizer.tokenize(document)])
        stemmed_documents.append(sentence)
    
    return np.array(stemmed_documents, dtype='object')


def encode_categorical_data(labels):
    return LabelEncoder().fit_transform(labels)


def preprocess_data(X, y, dataset_type, tf_idf):    
    X = stemming_documents(X)

    if dataset_type == 'train':
        X = tf_idf.fit_transform(X).toarray()
    elif dataset_type == 'test' or dataset_type == 'validation':
        X = tf_idf.transform(X).toarray()
           
    y = encode_categorical_data(y)
    y = y.reshape(-1, 1)

    return X, y


def k_fold_cross_validation(X, y, build_fn, k, batch, epochs):
    classifier = KerasClassifier(build_fn=build_fn, batch_size=batch, epochs=epochs)
    accuracies = cross_val_score(estimator=classifier, X=X, y=y, cv=k)
    mean = accuracies.mean()
    variance = accuracies.std()
    
    return mean, variance


def tokenize_dataset(tokenizer, X, y):
    X = tokenizer.texts_to_sequences(X)
    y = encode_categorical_data(y)
    y = y.reshape(-1, 1)
    
    return X, y


def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


def plot_history(acc, val_acc, loss, val_loss, file_name):
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Precisão de Treinamento')
    plt.plot(x, val_acc, 'r', label='Precisão de Validação')
    plt.title('Precisão de Treinamento e Validação')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Perda de Treinamento')
    plt.plot(x, val_loss, 'r', label='Perda de Validação')
    plt.title('Perda de Treinamento e Validação')
    plt.legend()
    
    plt.savefig('images/{}.png'.format(file_name))