import gui
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from others.popup import popup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import cross_validate
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss


def full_analysis(run):
    def pre_process():
        raw = pd.read_csv('dataset/articles.csv', encoding='latin-1')
        data = raw.Full_Article
        lb, type = raw.Article_Type.factorize()
        if run:
            final_rev = []
            "stop word and stemming is used for pre_processing"
            for idx, words in enumerate(data):
                print(idx)
                stop_words = set(stopwords.words('english'))
                rev_pr = [w for w in words.split() if not w in stop_words]
                stemmer = PorterStemmer()
                stemmed_rev = [stemmer.stem(word) for word in rev_pr]
                long_words_rev, long_words_revt = [], []
                for i in stemmed_rev:
                    if len(i) >= 3:  # removing short word
                        long_words_rev.append(i)
                final_rev.append((" ".join(long_words_rev)).strip())
            # np.save('pre_data', final_rev)
        pre_data = np.load('pre_evaluated/pre_data.npy', allow_pickle=True)
        return data, pre_data, lb

    def vectorization_(data):
        " Sentence bert is used "
        if run:
            model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
            embeddings = model.encode(data)
            # np.save('vect_data', embeddings)
        vect_data = np.load('pre_evaluated/vect_data.npy', allow_pickle=True)
        return vect_data

    def train(data, lb):
        global cr, pm
        if run:
            X_train, X_test, y_train, y_test = train_test_split(data, lb, test_size=0.2, random_state=42)
            # lstm_X_train = X_train.reshape(-1, X_train.shape[1], 1)
            # lstm_X_test = X_test.reshape(-1, X_test.shape[1], 1)
            #
            #
            # model = Sequential()
            # model.add(LSTM(128, input_shape=lstm_X_train[0].shape))
            # model.add(Dense(len(np.unique(lb)), activation='sigmoid'))
            # model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            # model.fit(lstm_X_train, y_train, epochs=10, batch_size=350, verbose=True)
            #
            # cross_ = cross_validate(model, data, lb, cv=5,
            #                         scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])
            #
            # y_predict = (model.predict_classes(lstm_X_test)).flatten()
            #
            # accuracy_score(y_test, y_predict)

            mdl = SVC(kernel="rbf",
                      C=3)  # rbf= 90 acc,c=5, sigmoid = 86acc, liner wit c =5 ==86acc, gamma = auto 0.83 gamma=scale=89
            mdl.fit(X_train, y_train)
            # joblib.dump(mdl, 'svm_model.pkl')
            pred = mdl.predict(X_test)

            scores = cross_validate(mdl, data, lb, cv=5,
                                    scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])
            cr = np.array(
                [scores['test_accuracy'], scores['test_precision_macro'], scores['test_recall_macro'],
                 scores['test_f1_macro']])
            # np.save('pre_evaluated/cross_results', cr)

            pm = np.array([accuracy_score(y_test, pred), precision_score(y_test, pred, average='weighted'),
                           recall_score(y_test, pred, average='weighted'),
                           f1_score(y_test, pred, average='weighted')])
        cr, pm = np.load('pre_evaluated/cross_results.npy', allow_pickle=True), np.load('pre_evaluated/pm.npy',
                                                                                        allow_pickle=True)
        return cr, pm

    data, pre_data, lb = pre_process()
    vect_data = vectorization_(pre_data)
    cr, pm = train(vect_data, lb)
    pm_df = pd.DataFrame(pm, columns=['Performance metrics'], index=['Accuracy', 'Precision', 'Recall', 'F1'])

    cr_df = pd.DataFrame(cr, columns=['k_fold1', 'k_fold2', 'k_fold3', 'k_fold4', 'k_fold5'],
                         index=['Accuracy', 'Precision', 'Recall', 'F1'])

    print('for hyper parameters such as KERNAL =rbf  is used because rbf kernal gives prediction better than the '
          'other kernal, and the C value is tuned to 5 for better accuracy')
    print('Cross-validation Results')
    print(cr_df.to_markdown())
    print('performance metrics Results')
    print(pm_df.to_markdown())


popup(full_analysis, full_analysis, gui.guii)
