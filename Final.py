import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv('stock_data.csv')
print(data.head(10))

data.Sentiment.value_counts()


import plotly.graph_objects as go
from plotly.offline import iplot
import plotly.express as px

fig = px.bar(x=data.Sentiment.unique(),y=[data.Sentiment.value_counts()],color=["1","-1"],text=data.Sentiment.value_counts())
fig.update_traces(hovertemplate="Sentiment:'%{x}' Counted: %{y}")
fig.update_layout(title={"text":"Sentiment Counts"},xaxis={"title":"Sentiment"},yaxis={"title":"Count"})
fig.show()

wordList = list()
for i in range(len(data)):
    temp = data.Text[i].split()
    for k in temp:
        wordList.append(k)


from collections import Counter
wordCounter = Counter(wordList)
countedWordDict = dict(wordCounter)
sortedWordDict = sorted(countedWordDict.items(),key = lambda x : x[1],reverse=True)
sortedWordDict[0:20]

num = 100
list1 = list()
list2 = list()
for i in range(num):
    list1.append(wordCounter.most_common(num)[i][0])
    list2.append(wordCounter.most_common(num)[i][1])

fig2 = px.bar(x=list1,y=list2,color=list2,hover_name=list1,hover_data={'Word':list1,"Count":list2})
fig2.update_traces(hovertemplate="Word:'%{x}' Counted: %{y}")
fig2.update_layout(title={"text":"Word Counts"},xaxis={"title":"Words"},yaxis={"title":"Count"})
fig2.show()

from wordcloud import WordCloud
from nltk.corpus import stopwords

wordList2 = " ".join(wordList)
stopwordCloud = set(stopwords.words("english"))
wordcloud = WordCloud(stopwords=stopwordCloud,max_words=2000,background_color="white",min_font_size=3).generate_from_frequencies(countedWordDict)
plt.figure(figsize=[13,10])
plt.axis("off")
plt.title("Most used words",fontsize=20)
plt.imshow(wordcloud)
plt.show()

# First of all, We need to change negative ones to zeros for our NN
print("***********Before************")
print(data.Sentiment.head(10))
data.Sentiment = data.Sentiment.replace(-1,0)
print("***********After*************")
print(data.Sentiment.head(10))
fig = px.bar(x=data.Sentiment.unique(),y=[data.Sentiment.value_counts()],color=["1","0"],text=data.Sentiment.value_counts())
fig.update_traces(hovertemplate="Sentiment:'%{x}' Counted: %{y}")
fig.update_layout(title={"text":"Sentiment Counts"},xaxis={"title":"Sentiment"},yaxis={"title":"Count"})
fig.show()


# Secondly, It's not very important but I wanna use same sizes of values due to overfitting
data2 = data.sort_values(by="Sentiment")
data2 = data2.reset_index().iloc[0:,1:3]
print("2105:",data2["Sentiment"][2105])
print("2106:",data2["Sentiment"][2106])
data3 = data2.iloc[0:2106*2]
print("New value counts")
print(data3.Sentiment.value_counts())
data = data3

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize, WordNetLemmatizer

ps = PorterStemmer()
lemma = WordNetLemmatizer()
stopwordSet = set(stopwords.words('english'))

# So let's print one by one to see what is going on
print("1)",data['Text'][0])
text = re.sub('[^a-zA-Z]'," ",data['Text'][0]) # clearing special characters and numbers
print("2)",text)
text = text.lower()                            # lower
print("3)",text)
text = word_tokenize(text,language='english')  # split
print("4)",text)
text1 = [word for word in text if not word in stopwordSet] #clearing stopwords like "to", "it", "over"
text2 = [lemma.lemmatize(word) for word in text]           #same thing
text = [lemma.lemmatize(word) for word in text if(word) not in stopwordSet] # I prefer using both but as you can see they are same
print("5.1)",text1)
print("5.2)",text2)
print("5)",text)
text = " ".join(text)                          # list -> string
print("6)",text)


textList = list()
for i in range(len(data)):
    text = re.sub('[^a-zA-Z]'," ",data['Text'][i])
    text = text.lower()
    text = word_tokenize(text,language='english')
    text = [lemma.lemmatize(word) for word in text if(word) not in stopwordSet]
    text = " ".join(text)
    textList.append(text)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

cv = CountVectorizer(max_features=5001)  # you can change max_features to see different results
x = cv.fit_transform(textList).toarray() # strings to 1 and 0
#cvs = x.sum(axis=0)
#print(cvs)          # to see word sum column by column

y = data["Sentiment"]

pca = PCA(n_components=256) # you can change n_components to see different results
x = pca.fit_transform(x)    # fits 5001 columns to 256 with minimal loss

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=21) # splitting x and y for train/test


from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

modelList = []
modelList.append(("LogisticReg",LogisticRegression()))
modelList.append(("GaussianNB",GaussianNB()))
modelList.append(("BernoulliNB",BernoulliNB()))
modelList.append(("DecisionTree",DecisionTreeClassifier()))
modelList.append(("RandomForest",RandomForestClassifier()))
modelList.append(("KNeighbors",KNeighborsClassifier(n_neighbors=5)))
modelList.append(("SVC",SVC()))
modelList.append(("XGB",XGBClassifier()))

def train_predict(x_train,x_test,y_train,y_test):
    for name, classifier in modelList:
        classifier.fit(x_train,y_train)
        y_pred = classifier.predict(x_test)
        print("{} Accuracy: {}".format(name,accuracy_score(y_test,y_pred)))

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model

def build_model():
    model = Sequential()
    
    model.add(Dense(units=16,activation="relu",init="uniform",input_dim=x.shape[1]))
    model.add(Dense(units=16,activation="relu",init="uniform"))
    model.add(Dense(units=1,activation="sigmoid",init="uniform"))
    
    optimizer = Adam(lr=0.0001,beta_1=0.9,beta_2=0.999)
    #optimizer = RMSprop(lr=0.0001,rho=0.9)
    
    model.compile(optimizer=optimizer,metrics=["accuracy"],loss="binary_crossentropy")
    return model

from keras.models import Sequential
from keras.layers import Dense

def build_model():
    model = Sequential()
    model.add(Dense(units=16, activation="relu", kernel_initializer="uniform", input_dim=x.shape[1]))
    model.add(Dense(units=16, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))
    return model

model = build_model()
model.summary()


import tensorflow as tf


optimizer = tf.keras.optimizers.Adam(learning_rate=5e-05, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
metric = tf.keras.metrics.CategoricalAccuracy('balanced_accuracy')

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[metric]
)



model.fit(x_train,y_train,epochs=15,verbose=1)
import numpy as np

y_pred_proba = model.predict(x_test)
y_pred3 = np.argmax(y_pred_proba, axis=1)



train_predict(x_train,x_test,y_train,y_test)
print("ANN Accuracy: ",accuracy_score(y_test,y_pred3.ravel()))
print("ANN Confusion Matrix")
print(confusion_matrix(y_test,y_pred3.ravel()))


import joblib

# Save the model
joblib.dump(model, 'model.joblib')

# Load the model
loaded_model = joblib.load('model.joblib')


# Assuming loaded_model is already loaded using joblib.load()

def preprocess_input(single_input):
    text = re.sub('[^a-zA-Z]', " ", single_input)
    text = text.lower()
    text = word_tokenize(text, language='english')
    text = [lemma.lemmatize(word) for word in text if word not in stopwordSet]
    text = " ".join(text)
    return text

def vectorize_input(input_text):
    text_list = [input_text]
    x_input = cv.transform(text_list).toarray()
    x_input = pca.transform(x_input) if 'pca' in globals() else x_input
    return x_input

def make_prediction(single_input):
    preprocessed_input = preprocess_input(single_input)
    vectorized_input = vectorize_input(preprocessed_input)
    prediction = loaded_model.predict(vectorized_input)
    return prediction

# Example usage:
single_input_text = "Bitcoin huge increase gain gain ETFs Gain  goog good good good in Asia Post U.S. Approval"
prediction_result = make_prediction(single_input_text)
print("Predicted Sentiment:", prediction_result)