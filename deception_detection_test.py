import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dropout, Dense, Conv1D, MaxPooling1D, Flatten, Embedding, GlobalMaxPool1D
from keras import optimizers
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.constraints import maxnorm

from deception_detection import concat_datasets, simplify_labels, split_dataframe, stemming_documents, encode_categorical_data, preprocess_data, k_fold_cross_validation, tokenize_dataset, create_embedding_matrix, plot_history


def execute_ann_tf_idf(mode='cv'):
    df = concat_datasets(pd.read_csv('datasets/train.tsv', sep='\t'),
                pd.read_csv('datasets/test.tsv', sep='\t'),
                pd.read_csv('datasets/valid.tsv', sep='\t'))
    X, y = split_dataframe(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    tf_idf = TfidfVectorizer(max_df=0.5)

    X_train, y_train = preprocess_data(X_train, y_train, 'train', tf_idf)
    X_test, y_test = preprocess_data(X_test, y_test, 'test', tf_idf)

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
    
    if mode == 'cv':
    	accuracy, variance = k_fold_cross_validation(X_train, y_train, build_ann_classifier, 5, 50, 3)
    	print('Accuracy: {}\nVariance: {}\n'.format(accuracy, variance))
    else:
        acc = []
        val_acc = []
        loss = []
        val_loss = []
        test_acc = []
        
        for i in range(30):
            classifier = build_ann_classifier()

            history = classifier.fit(X_train, y_train,
                                epochs=3,
                                validation_split=0.2,
                                batch_size=50,
                                verbose=0)
            
            acc_v = history.history['acc']
            val_acc_v = history.history['val_acc']
            loss_v = history.history['loss']
            val_loss_v = history.history['val_loss']
            
            plot_history(acc_v, val_acc_v, loss_v, val_loss_v, 'acc_loss_ann_{}'.format(i))

            acc.append(acc_v)
            val_acc.append(val_acc_v)
            loss.append(loss_v)
            val_loss.append(val_loss_v)

            test_acc.append(classifier.evaluate(X_test, y_test)[1])

        acc = np.array(acc).mean(axis=0)
        val_acc = np.array(val_acc).mean(axis=0)
        loss = np.array(loss).mean(axis=0)
        val_loss = np.array(val_loss).mean(axis=0)
        
        plot_history(acc, val_acc, loss, val_loss, 'acc_loss_ann_average')

        print('ANN')
        print(test_acc)


def execute_ann_tokenized(mode='cv'):
    df = concat_datasets(pd.read_csv('datasets/train.tsv', sep='\t'),
                pd.read_csv('datasets/test.tsv', sep='\t'),
                pd.read_csv('datasets/valid.tsv', sep='\t'))

    X, y = split_dataframe(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1000)

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)

    X_train, y_train = tokenize_dataset(tokenizer, X_train, y_train)
    X_test, y_test = tokenize_dataset(tokenizer, X_test, y_test)
    max_len = 100
    
    X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
    X_test = pad_sequences(X_test, padding='post', maxlen=max_len)

    embedding_matrix = create_embedding_matrix('datasets/glove.6B.50d.txt', tokenizer.word_index, 50)

    def build_ann_classifier_tokenized():
        classifier = Sequential()
        classifier.add(Dense(10, kernel_initializer='uniform', activation='relu', input_dim=100))
        classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

        classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

        return classifier

    if mode == 'cv':
    	accuracy, variance = k_fold_cross_validation(X_train, y_train, build_ann_classifier_tokenized, 5, 128, 10)
    	print('Accuracy: {}\nVariance: {}\n'.format(accuracy, variance))
    else:
        acc = []
        val_acc = []
        loss = []
        val_loss = []
        test_acc = []
        
        for i in range(30):
            classifier = build_ann_classifier_tokenized()

            history = classifier.fit(X_train, y_train,
                                epochs=10,
                                validation_split=0.2,
                                batch_size=128,
                                verbose=0)
            
            acc_v = history.history['acc']
            val_acc_v = history.history['val_acc']
            loss_v = history.history['loss']
            val_loss_v = history.history['val_loss']
            
            plot_history(acc_v, val_acc_v, loss_v, val_loss_v, 'acc_loss_ann_tokenized_{}'.format(i))

            acc.append(acc_v)
            val_acc.append(val_acc_v)
            loss.append(loss_v)
            val_loss.append(val_loss_v)

            test_acc.append(classifier.evaluate(X_test, y_test)[1])

        acc = np.array(acc).mean(axis=0)
        val_acc = np.array(val_acc).mean(axis=0)
        loss = np.array(loss).mean(axis=0)
        val_loss = np.array(val_loss).mean(axis=0)
        
        plot_history(acc, val_acc, loss, val_loss, 'acc_loss_ann_tokenized_average')

        print('ANN tokenized')
        print(test_acc)


def execute_cnn(mode='cv'):
    df = concat_datasets(pd.read_csv('datasets/train.tsv', sep='\t'),
                pd.read_csv('datasets/test.tsv', sep='\t'),
                pd.read_csv('datasets/valid.tsv', sep='\t'))

    X, y = split_dataframe(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1000)

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)

    X_train, y_train = tokenize_dataset(tokenizer, X_train, y_train)
    X_test, y_test = tokenize_dataset(tokenizer, X_test, y_test)

    vocab_size = len(tokenizer.word_index) + 1
    max_len = 100
    
    X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
    X_test = pad_sequences(X_test, padding='post', maxlen=max_len)

    embedding_matrix = create_embedding_matrix('datasets/glove.6B.50d.txt', tokenizer.word_index, 50)

    def build_cnn_classifier():
        classifier = Sequential()
        classifier.add(Embedding(input_dim=12500, output_dim=50, weights=[embedding_matrix], input_length=100, trainable=True))
        classifier.add(Dropout(0.3))
        classifier.add(Conv1D(64, 2, kernel_initializer='uniform', activation='relu'))
        classifier.add(GlobalMaxPool1D())
        classifier.add(Dense(10, kernel_initializer='uniform', activation='relu', kernel_constraint=maxnorm(3)))
        classifier.add(Dropout(0.4))
        classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

        classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

        return classifier
    
    if mode == 'cv':
        accuracy, variance = k_fold_cross_validation(X_train, y_train, build_cnn_classifier, 5, 128, 20)
        print('Accuracy: {}\nVariance: {}\n'.format(accuracy, variance))
    else:
        acc = []
        val_acc = []
        loss = []
        val_loss = []
        test_acc = []
        
        for i in range(30):
            classifier = build_cnn_classifier()

            history = classifier.fit(X_train, y_train,
                                epochs=10,
                                validation_split=0.2,
                                batch_size=128,
                                verbose=0)
            
            acc_v = history.history['acc']
            val_acc_v = history.history['val_acc']
            loss_v = history.history['loss']
            val_loss_v = history.history['val_loss']
            
            plot_history(acc_v, val_acc_v, loss_v, val_loss_v, 'acc_loss_cnn_{}_tuned'.format(i))

            acc.append(acc_v)
            val_acc.append(val_acc_v)
            loss.append(loss_v)
            val_loss.append(val_loss_v)

            test_acc.append(classifier.evaluate(X_test, y_test)[1])

        acc = np.array(acc).mean(axis=0)
        val_acc = np.array(val_acc).mean(axis=0)
        loss = np.array(loss).mean(axis=0)
        val_loss = np.array(val_loss).mean(axis=0)
        
        plot_history(acc, val_acc, loss, val_loss, 'acc_loss_cnn_average_tuned')

        print('CNN')
        print(test_acc)

        
if __name__ == '__main__':
    # execute_ann_tf_idf('exp')
    # execute_ann_tokenized('exp')
    execute_cnn('exp')