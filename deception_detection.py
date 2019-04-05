import numpy as np
import pandas as pd
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Dense, Conv1D, MaxPooling1D, Flatten
from keras.wrappers.scikit_learn import KerasClassifier


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


def build_ann_classifier(rate=0.1, lr=0.01):
    classifier = Sequential()
    
    classifier.add(Dense(units=512, kernel_initializer='uniform', activation='relu', input_shape=(10229,)))
    classifier.add(Dropout(rate=rate))
    classifier.add(Dense(units=512, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(rate=rate))
    classifier.add(Dense(units=256, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(rate=rate))
    classifier.add(Dense(units=128, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(rate=rate))
    classifier.add(Dense(units=64, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=32, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    rmsprop = optimizers.RMSprop(lr=lr)
    classifier.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])

    return classifier


def predict_news(news):
    news = np.array(news)
    news = tf_idf.transform(news).toarray()
    
    return classifier.predict(news) 


def k_fold_cross_validation(X, y, build_fn, k, batch, epochs):
    classifier = KerasClassifier(build_fn=build_fn, batch_size=batch, epochs=epochs)
    accuracies = cross_val_score(estimator=classifier, X=X, y=y, cv=k)
    mean = accuracies.mean()
    variance = accuracies.std()
    
    return mean, variance


if __name__ == '__main__':
	df = concat_datasets(pd.read_csv('datasets/train.tsv', sep='\t'),
                pd.read_csv('datasets/test.tsv', sep='\t'),
                pd.read_csv('datasets/valid.tsv', sep='\t'))
	X, y = split_dataframe(df)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
	tf_idf = TfidfVectorizer(max_df=0.5)

	X_train, y_train = preprocess_data(X_train, y_train, 'train', tf_idf)
	X_test, y_test = preprocess_data(X_test, y_test, 'test', tf_idf)

	accuracy, variance = k_fold_cross_validation(X_train, y_train, build_ann_classifier, 5, 50, 3)

	print('Accuracy: {}\nVariance: {}\n'.format(accuracy, variance))
