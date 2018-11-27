import io
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn.metrics import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

class Model():
    def __init__(self):
        data = pd.read_pickle('sentiment.pkl')
        data['polarity'] = data['polarity'].map(lambda x: 0 if x == 0 else 1)
        self.train = data[:5000]
        self.test = data[5000:]
        self.vectorizer = CountVectorizer(token_pattern="\\w+", lowercase=True)
        self.vectorizer.fit(data.tweet.values)
        data = None
        self.trained = 'None'
        self.model = None
        self.trainX = None
        self.trainy = None
        self.testX = None
        self.testy = None
        
    def train_w2v(self):
        self.trainX = self.train.w2v.values
        for i in range(len(self.trainX)):
            if not isinstance(self.trainX[i], np.ndarray):
                self.trainX[i] = np.zeros(300, dtype='f4')
        self.trainX = np.array(self.trainX.tolist())
        self.trainy = self.train.polarity.values
        self.testX = self.test.w2v.values
        for i in range(len(self.testX)):
            if not isinstance(self.testX[i], np.ndarray):
                self.testX[i] = np.zeros(300, dtype='f4')
        self.testX = np.array(self.testX.tolist())
        self.testy = self.test.polarity.values
        self.model = linear_model.LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)
        self.model.fit(self.trainX, self.trainy)
        self.trained = 'Word 2 Vec'
        
    def train_onehot(self):
        self.trainX = self.vectorizer.transform(self.train.tweet.values)
        self.trainy = self.train.polarity.values
        self.testX = self.vectorizer.transform(self.test.tweet.values)
        self.testy = self.test.polarity.values
        self.model = linear_model.LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)
        self.model.fit(self.trainX, self.trainy)
        self.trained = 'One-Hot'
    
    def test_model(self):
        return accuracy_score(self.model.predict(self.testX), self.testy)

    def plot_roc(self):
        x, y, _ = roc_curve(self.testy, self.model.predict(self.testX))
        my_auc = auc(x, y)
        fig = Figure()
        axis = fig.add_subplot(1, 1, 1)
        lw = 2
        axis.plot(x, y, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % my_auc)
        axis.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        return fig
