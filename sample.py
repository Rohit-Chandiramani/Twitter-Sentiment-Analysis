import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
dataset=pd.read_csv("train.csv",sep=',',encoding='latin1')
corpus=[]
accuracies=[]
for i in range(0,30000):
    tweet=re.sub('[^a-zA-Z]',' ',dataset['SentimentText'][i])
    tweet=tweet.lower()
    tweet=tweet.split()
    ps=PorterStemmer()
    lemmatizer=WordNetLemmatizer()
    tweet=[lemmatizer.lemmatize(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet=' '.join(tweet)
    corpus.append(tweet)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.70, min_df=2, max_features=7000, stop_words='english')
#from sklearn.feature_extraction.text import CountVectorizer
#cv=CountVectorizer(max_features=2000)
X=tfidf_vectorizer.fit_transform(corpus).toarray()
y=dataset.iloc[0:30000,1].values



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)


#tfidf_vectorizer.get_feature_names()


print("Gaussian Naive Bayes")
from sklearn.naive_bayes import GaussianNB
classifier_GNB=GaussianNB()
classifier_GNB.fit(X_train,y_train)

pred=classifier_GNB.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,pred)
print(cm)
from sklearn.metrics import accuracy_score
print('Accuracy of model is :- '+str(accuracy_score(y_test,pred)))
accuracies.append(accuracy_score(y_test,pred))



print("Multinomial Naive Bayes")


from sklearn.naive_bayes import MultinomialNB
classifier_MNB=MultinomialNB()
classifier_MNB.fit(X_train,y_train)
pred=classifier_MNB.predict(X_test)
cm=confusion_matrix(y_test,pred)
print(cm)
print('Accuracy of model is :- '+str(accuracy_score(y_test,pred)))
accuracies.append(accuracy_score(y_test,pred))



print("Logistic Regression")
from sklearn.linear_model import LogisticRegression
classifier_logistic=LogisticRegression()
classifier_logistic.fit(X_train,y_train)

pred=classifier_logistic.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,pred)
print(cm)


from sklearn.metrics import accuracy_score
print('Accuracy of model is :- '+str(accuracy_score(y_test,pred)))
accuracies.append(accuracy_score(y_test,pred))


print("Linear SVC")
from sklearn.svm import LinearSVC
classifier_svc=LinearSVC()
classifier_svc.fit(X_train,y_train)

pred=classifier_svc.predict(X_test)

cm=confusion_matrix(y_test,pred)
print(cm)


print('Accuracy of model is :- '+str(accuracy_score(y_test,pred)))
accuracies.append(accuracy_score(y_test,pred))


print("ensemble classifier")
from sklearn.ensemble import VotingClassifier
estimators=[]
estimators.append(('MultinomialNB',classifier_MNB))
estimators.append(('LogisticRegression',classifier_logistic))
estimators.append(('LinearSVC',classifier_svc))


ensemble_classifier=VotingClassifier(estimators)
ensemble_classifier.fit(X_train,y_train)
pred=ensemble_classifier.predict(X_test)
cm=confusion_matrix(y_test,pred)
print(cm)


print('Accuracy of model is :- '+str(accuracy_score(y_test,pred)))
accuracies.append(accuracy_score(y_test,pred))


#final histogram
classifier_types =['GNB','MNB','LRG','LinearSVC','Ensemble']
tp = [1,2,3,4,5]
for i in range (0,5):
    accuracies[i]=accuracies[i]*100
    accuracies[i]=accuracies[i].round(4)
 

plt.bar(tp,accuracies,color='red')
plt.yticks(np.arange(0,110,10))
plt.xticks(tp,classifier_types)
plt.xlabel('Classifier')
plt.ylabel('% Accuracy')
plt.title('Accuracy Comparision')
plt.savefig('fig1.png')
plt.legend()
plt.show()

print(accuracies)


#accuracies on various samples recorded 
'''
Stemming
[65.0, 67.5, 65.0, 62.5, 67.5]  -200
[59.0, 79.0, 79.0, 74.0, 80.0]  -500
[63.5, 74.0, 73.0, 71.0, 73.0]  -1000
[54.50, 79.20, 78.79, 78.10, 80.5] -5000
[56.85, 76.04, 76.95, 75.40, 77.0] -10000
[61.63, 73.33, 73.63, 72.66, 73.45]  -30000
[62.42, 73.12, 73.62, 72.95, 73.56]  -50000
[62.35, 72.67, 73.73, 72.82, 73.44]  -70000
[56.41, 73.15, 74.22, 73.47, 73.87]  -90000


Lemmatizing
[57.5, 57.5, 62.5, 52.5, 57.5]  -200
[56.99, 80.0, 79.0, 76.0, 83.0]  -500
[62.5, 74.5, 74.0, 72.5, 74.0]  -1000
[56.69, 78.29, 77.70, 77.40, 79.10]  -5000
[55.95, 76.20, 76.15, 75.20, 76.54]  -10000
[62.36, 72.83, 73.21, 72.21, 73.18]  -30000
[63.77, 72.93, 73.0, 72.62, 73.15]  -50000
[64.94, 73.82, 74.02, 73.67, 73.921]  -70000
[58.33, 73.15, 74.09, 73.53, 73.91]  -9000



taking only ensemble accuracies as measure for visualization
'''

Xplot=[200,500,1000,5000,10000,30000,50000,70000,90000]
yplot=[67.5,80.0,73.0,80.5,77.0,73.45,73.56,73.44,73.87]


plt.scatter(Xplot,yplot,color='red')
plt.plot(Xplot,yplot)
plt.savefig('fig2.png')
plt.xlabel('No. of tweets')
plt.ylabel('%Accuracy')
plt.title('No. of tweets VS %Accuracy')

'''
for i in range(0,9):
    plt.text(Xplot[i],yplot[i],'('+str(Xplot[i])+','+str(yplot[i])+')',size=8)
    
'''
plt.savefig('fig2.png')

plt.legend()
plt.show()
