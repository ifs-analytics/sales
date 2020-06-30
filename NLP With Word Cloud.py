import string
import re
import numpy as np
import pandas as pd
import matplotlib
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)


with open(r"C:\Users\niko\PycharmProjects\untitled2\sentiment_labelled_sentences\full_set.txt", 'r',encoding="utf8") as f:
    content = f.readlines()

#Preprocessing

## Remove leading and trailing white space
content = [x.strip() for x in content]
## Separate the sentences from the labels
sentences = [x.split("\t")[0] for x in content]
labels = [x.split("\t")[1] for x in content]

## Transform the labels from '0 v.s. 1' to '-1 v.s. 1'
y = np.array(labels, dtype='int8')
y = 2*y - 1

def full_remove(x, removal_list):
    for w in removal_list:
        x = x.replace(w, ' ')
    return x
## Remove digits ##
digits = [str(x) for x in range(10)]
remove_digits = [full_remove(x, digits) for x in sentences]
## Remove punctuation ##
remove_punc = [full_remove(x, list(string.punctuation)) for x in remove_digits]
## Make everything lower-case and remove any white space ##
sents_lower = [x.lower() for x in remove_punc]
sents_lower = [x.strip() for x in sents_lower]


## Remove stop words ##
stop_set = ['the', 'a', 'an', 'i', 'he', 'she', 'they', 'to', 'of', 'it', 'from']
def removeStopWords(stop_set, txt):
    newtxt = ' '.join([word for word in txt.split() if word not in stop_set])
    return newtxt
sents_processed = [removeStopWords(stop_set,x) for x in sents_lower]

#bag of words creation
vectorizer = CountVectorizer(analyzer = "word",
                             preprocessor = None,
                             stop_words =  'english',
                             max_features = 6000, ngram_range=(1,5))
data_features = vectorizer.fit_transform(sents_processed)
tfidf_transformer = TfidfTransformer()
data_features_tfidf = tfidf_transformer.fit_transform(data_features)
data_mat = data_features_tfidf.toarray()

#data split
np.random.seed(0)
test_index = np.append(np.random.choice((np.where(y==-1))[0], 250, replace=False), np.random.choice((np.where(y==1))[0], 250, replace=False))
train_index = list(set(range(len(labels))) - set(test_index))
train_data = data_mat[train_index,]
train_labels = y[train_index]
test_data = data_mat[test_index,]
test_labels = y[test_index]

#Create polarity function and subjectivity function
pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity
pol_list = [pol(x) for x in sents_processed]
sub_list = [sub(x) for x in sents_processed]

## Fit logistic classifier on training data
clf = SGDClassifier(loss="log", penalty="none")
clf.fit(train_data, train_labels)
## Pull out the parameters (w,b) of the logistic regression model
w = clf.coef_[0,:]
b = clf.intercept_
## Get predictions on training and test data
preds_train = clf.predict(train_data)
preds_test = clf.predict(test_data)
## Compute errors
errs_train = np.sum((preds_train > 0.0) != (train_labels > 0.0))
errs_test = np.sum((preds_test > 0.0) != (test_labels > 0.0))
##print("Training error: ", float(errs_train)/len(train_labels))
##print("Test error: ", float(errs_test)/len(test_labels))


##LSTM network
max_review_length = 200
tokenizer = Tokenizer(num_words=10000,  #max no. of unique words to keep
                      filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
                      lower=True #convert to lower case
                     )
tokenizer.fit_on_texts(sents_processed)

X = tokenizer.texts_to_sequences(sents_processed)
X = sequence.pad_sequences(X, maxlen= max_review_length)

Y=pd.get_dummies(y).values

np.random.seed(0)
test_inds = np.append(np.random.choice((np.where(y==-1))[0], 250, replace=False), np.random.choice((np.where(y==1))[0], 250, replace=False))
train_inds = list(set(range(len(labels))) - set(test_inds))
train_data = X[train_inds,]
train_labels = Y[train_inds]
test_data = X[test_inds,]
test_labels = Y[test_inds]


##network Creation
EMBEDDING_DIM = 200
model = Sequential()
model.add(Embedding(10000, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(250, dropout=0.2,return_sequences=True))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 2
batch_size = 40
model.fit(train_data, train_labels,
          epochs=epochs,
          batch_size=batch_size,
          validation_split=0.1,
          verbose=0)


###testing
outcome_labels = ['Negative', 'Positive']
new = ["Tresl is a great place to shop for loans"]

seq = tokenizer.texts_to_sequences(new)
padded = sequence.pad_sequences(seq, maxlen=max_review_length)
pred = model.predict(padded)
print("Probability distribution: ", pred)
print("Is this a Positive or Negative review? ")
print(outcome_labels[np.argmax(pred)])

## High scoring Positive/Negative Words

## Convert vocabulary into a list:
vocab = np.array([z[0] for z in sorted(vectorizer.vocabulary_.items(), key=lambda x:x[1])])
## Get indices of sorting w
inds = np.argsort(w)
## Words with large negative values
neg_inds = inds[0:50]
#print("Highly negative words: ")
#print([str(x) for x in list(vocab[neg_inds])])
## Words with large positive values
pos_inds = inds[-49:-1]
#print("Highly positive words: ")
#print([str(x) for x in list(vocab[pos_inds])])



###Word Cloud

##from wordcloud import WordCloud
#wc = WordCloud(stopwords=stop_set, background_color="white", colormap="Dark2",
               #max_font_size=150, random_state=42)
#plt.rcParams['figure.figsize'] = [16, 6]
#wc.generate(" ".join(list(vocab[neg_inds])))
#plt.imshow(wc, interpolation="bilinear")
#plt.axis("off")#plt.show()