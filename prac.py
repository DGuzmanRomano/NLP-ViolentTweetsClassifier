from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import os, pickle
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import spacy
from sklearn.model_selection import train_test_split
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy import displacy
from numpy import dot
from numpy.linalg import norm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import NMF
from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import SparsePCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from unidecode import unidecode

import sys
#np.set_printoptions(threshold=sys.maxsize)

class data_set_attraction:
	def __init__(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test


def lemmatization(train):
	lemmatized_train = []

	stoppos = ["ADV","ADP","DET","PRON","SCONJ","CCONJ","PUNCT"]

	nlp = spacy.load("es_core_news_sm")

	for index in range(len(train)):
		
		line = train[index]
		string = str(line)
		lemmatized_string = ''

		cleaned_text = re.sub(r"(\[)|(\d)|(\\)|(xa0)|(\b\d\w*\b)|https?://t\.co/\S+", "", string)
		
		doc = nlp(cleaned_text)
	
		for token in doc:
			
			if((token.pos_ not in stoppos)):
				
				token.lemma_ = unidecode(token.lemma_)
				
				lemmatized_string = lemmatized_string + token.lemma_ + " "
				


		
			
		lemmatized_string = lemmatized_string[:-1]
		
		lemmatized_train.append(lemmatized_string)

		
	return (lemmatized_train)




def train_model(corpus):
	
	# Representación vectorial por frecuencia
	vectorizer = CountVectorizer()
	#vectorizer =TfidfVectorizer()
	

	X_train = vectorizer.fit_transform(corpus.X_train)
	
	#pca = SparsePCA(n_components=5000, alpha=0.1)
	#X_train = pca.fit_transform(X_train.toarray())
	
	y_train = corpus.y_train
	print (vectorizer.get_feature_names_out())
	print (X_train.shape)
	y_train = y_train.ravel()
	
	print("************************")
	classifiers = [
    {'name': 'RandomForestClassifier', 'estimator': RandomForestClassifier(), 'params': {'n_estimators': [10, 50, 100], 'max_depth': [None, 5, 10]}},
    {'name': 'SVC', 'estimator': SVC(), 'params': {'C': [0.1, 1, 10,20], 'kernel': ['linear', 'rbf'],'gamma': [0.1, 1, 'scale',0.01]}},
    {'name': 'KNN', 'estimator': KNeighborsClassifier(), 'params': {'n_neighbors': [3, 5, 10,20,30,50],'weights': ['uniform', 'distance'],'p': [1, 2] } },
    {'name': 'DecisionTreeClassifier', 'estimator': DecisionTreeClassifier(), 'params': {'max_depth': [2, 4, 6, 8, 10,20,50],'min_samples_leaf': [1, 2, 4, 8],'min_samples_split': [2, 4, 8,10] } },
	{'name': 'MultinomialNB', 'estimator': MultinomialNB(), 'params': {'alpha': [0.1, 1.0,2,3],'fit_prior': [True, False] } }
	
]
	
	best_estimator = None
	best_score = -np.inf
	best_tp = -np.inf
	best_tn = -np.inf 
	best_cm = None
	best_classifiers = []


	for classifier in classifiers:
		
		print("Training", classifier['name'])
		grid_search = GridSearchCV(classifier['estimator'], classifier['params'], cv=5, scoring='accuracy', n_jobs=10)
		grid_search.fit(X_train, y_train)
		score = grid_search.best_score_
		print("Best parameters:", grid_search.best_params_)
		print("Best score:", score)
		best_classifier = grid_search.best_estimator_
		best_classifiers.append((classifier['name'], best_classifier))


		cm = mid_test(corpus, grid_search, vectorizer)
		print("Confusion matrix:\n", cm)
		tn, fp, fn, tp = cm.ravel()
		if tp > best_tp:
			best_tp = tp
			best_tp_classifier = classifier['name']
			st1_clf = classifier['estimator']
			best_tp_cm = cm
		if tn > best_tn:
			best_tn = tn
			best_tn_classifier = classifier['name']
			st2_clf = classifier['estimator']
			best_tn_cm = cm
			
        
			
		if score > best_score:
			best_score = score
			best_estimator = grid_search.best_estimator_
	print("Best estimator:", best_estimator)
	print("Classifier with most true positives:", best_tp_classifier)
	print("Confusion matrix with most true positives:\n", best_tp_cm)
	print("Classifier with most true negatives:", best_tn_classifier)
	print("Confusion matrix with most true negatives:\n", best_tn_cm)
	if best_tp_classifier != best_tn_classifier:
		print("Stacking")
		meta = LogisticRegression(max_iter=1540)
		stacking = StackingClassifier(estimators=[('lr', LogisticRegression(max_iter=1540)),('rf', st1_clf), ('svm', st2_clf)], 
                              final_estimator=meta,
                              cv=5,n_jobs=10)
		stacking.fit(X_train, y_train)
		test_model(corpus, stacking, vectorizer)
	print("BEST no stack")
	test_model(corpus, best_estimator, vectorizer)


	print("Voting")
	voting_clf = VotingClassifier(estimators=best_classifiers, voting='hard')	
	voting_clf.fit(X_train, y_train)
	test_model(corpus, voting_clf, vectorizer)

	print("BStacking")
	meta = LogisticRegression(max_iter=1540)
	stacking = StackingClassifier(estimators=[('lr', LogisticRegression(max_iter=1540)),
					   ('rf', RandomForestClassifier(max_depth=5, n_estimators=150)),
					     ('svm', SVC(C=35,gamma='scale',kernel='rbf')),
						 ('mnb', MultinomialNB(alpha=2))], 
                              final_estimator=meta,
                              cv=5,n_jobs=15)
	stacking.fit(X_train, y_train)
	test_model(corpus, stacking, vectorizer)
	















		
	#print("Test score:", best_estimator.score(X_test, y_test))
	#print("Confusion matrix:\n", confusion_matrix(y_test, best_estimator.predict(X_test)))
    
    
	#Se retorna el clasificador y el vectorizador
 




def mid_test(corpus, model, vectorizer):
	
	
	X_test = vectorizer.transform (corpus.X_test)
	




	y_test = corpus.y_test
	predictions = model.predict(X_test)
	return confusion_matrix(y_test, predictions)
	


def test_model(corpus, model, vectorizer):
	#Se vectoriza el conjunto de prueba
	X_test = vectorizer.transform (corpus.X_test)




	y_test = corpus.y_test
	
	#Se realizan las predicciones del conjunto de pruebas con el modelo entrenado
	predictions = model.predict(X_test)
	

	print (predictions)
	
	#Se calcula la exactitud del las predicciones del modelo entrenado
	print (accuracy_score(y_test, predictions))
	print(classification_report(y_test, predictions))



if __name__=='__main__':

    df = pd.read_csv('train_data.csv', sep=',',header=None,names=["code", "text"], engine='python')
    
    df.drop("code", inplace=True, axis=1)
  
    X=df.values

    X=lemmatization(X)

    df = pd.read_csv('train_labels_subtask_1.csv', sep=',',header=None,names=["value"], engine='python')
    #print (df)

    y=df.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle = True, random_state = 0 ) 

    #lemmatized_X_train, lemmatized_X_test = lemmatization(X_train, X_test)
    corpus = data_set_attraction(X_train, y_train,X_test, y_test)


    train_model(corpus)
        
    #Se prueba el modelo entrenado
    

