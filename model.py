import pandas as pd
import numpy as np
import pickle
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Ouverture du fichier 
df = pd.read_csv("data/labels.csv",",",encoding='utf-8', low_memory=False)

# Preprocessing de la data
df['tweet_tab_message'] = [[word for word in tweet.split(' ')] for tweet in df['tweet']]
df['tweet_message_treat'] = [[''.join(e for e in word if e.isalnum()) for word in tweet.split(' ') if word.find("@") == -1 and word.find("RT")] for tweet in df['tweet']]
df['tweet_tab_tag'] = [[word for word in tweet.split(' ') if word.find("@") != -1] for tweet in df['tweet']]
df['tweet_tab_message_with_out_special_char'] = [[''.join(e for e in word if e.isalnum()) for word in tweet.split(' ') if word.find("@") == -1] for tweet in df['tweet']]
df['tweet_message'] = df['tweet_message_treat'].apply(' '.join)

# Génération de modèle Linear SVC
X = df[['hate_speech','offensive_language','neither']]
y = df['class']

pipline = make_pipeline(StandardScaler(),LinearSVC(random_state=0, tol=1e-5))

pipline.fit(X, y)

pickle.dump(pipline.named_steps['linearsvc'], open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))