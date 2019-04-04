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


def split_dataframe(dataframe):
    dataframe['Label'] = dataframe['Label'].replace(['pants-fire', 'barely-true'], 'false').replace(['half-true', 'mostly-true'], 'true')
    
    X = dataframe.iloc[:, 2].values
    y = dataframe.iloc[:, 1].values
    
    return X, y


def encode_categorical_data(dataframe):
    return LabelEncoder().fit_transform(dataframe)


def stemming_documents(documents):
    whitespace_tokenizer = WhitespaceTokenizer()
    stemmer = PorterStemmer()
    stemmed_documents = []
    
    for document in documents:
        sentence = ' '.join([stemmer.stem(word).lower() for word in whitespace_tokenizer.tokenize(document)])
        stemmed_documents.append(sentence)
    
    return np.array(stemmed_documents, dtype='object')


def preprocess_datasets(X, y, dataset_type, tf_idf):    
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
    
    classifier.add(Dense(units=512, kernel_initializer='uniform', activation='relu', input_shape=(10230,)))
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

#    adam = optimizers.Adam(lr=lr)
#    classifier.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    rmsprop = optimizers.RMSprop(lr=lr)
    classifier.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])

    return classifier


def concat_datasets(train, test, validation):
    COLUMNS_LABELS = ['ID', 'Label', 'Statement', 'Subject', 'Speaker', "Speaker's job",
                     'State info', 'Party affiliation', 'Barely true counts', 'False counts',
                     'Half true counts', 'Mostly true counts', 'Pants on fire counts', 'Venue']
    
    train.columns = test.columns = validation.columns = COLUMNS_LABELS
    
    return pd.concat([train, test, validation])


def k_fold_cross_validation(X, y, build_fn, k, batch, epochs):
    classifier = KerasClassifier(build_fn=build_fn, batch_size=batch, epochs=epochs)
    accuracies = cross_val_score(estimator=classifier, X=X, y=y, cv=k)
    mean = accuracies.mean()
    variance = accuracies.std()
    
    return mean, variance


def find_optimal_hyperparameters(X, y, k, params):
    classifier = KerasClassifier(build_fn=build_ann_classifier)
    grid_search = GridSearchCV(estimator=classifier, param_grid=params, scoring='accuracy', cv=k)
    
    return grid_search.fit(X, y)


dataset = concat_datasets(pd.read_csv('train.tsv', sep='\t'),
                          pd.read_csv('test.tsv', sep='\t'),
                          pd.read_csv('valid.tsv', sep='\t'))
X, y = split_dataframe(dataset)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
tf_idf = TfidfVectorizer(max_df=0.5, ngram_range=(2, 2))

X_train, y_train = preprocess_datasets(X_train, y_train, 'train', tf_idf)
X_test, y_test = preprocess_datasets(X_test, y_test, 'test', tf_idf)

# uncomment for simple trainning
classifier = build_ann_classifier()
classifier.fit(X_train, y_train, batch_size=50, epochs=2, shuffle=True, validation_split=0.25)

# uncomment for CV
# accuracy, variance = k_fold_cross_validation(X_train, y_train, build_ann_classifier, 5, 50, 2)
# best acc = 0.6125530848050055, var = 0.0033601900549687412 rmsprop

# uncomment to predict a unique news
# news = np.array(['Our debt to GDP ratios, our deficit to GDP ratios are quickly approaching the countries we have been reading about for the last year and a half. We are not far behind Ireland, Greece, Portugal and Spain and all these countries we have been reading about.'])
# news = tf_idf.transform(news).toarray()
# response = classifier.predict(news) 

# uncomment for parameter tuning
#parameters = {'batch_size': [32, 50, 128, 256],
#              'epochs': [2, 3],
#              'lr': [0.001, 0.01, 0.1]}
#grid_search = find_optimal_hyperparameters(X_train, y_train, 5, parameters)
#best_parameters = grid_search.best_params_
#best_accuracy = grid_search.best_score_