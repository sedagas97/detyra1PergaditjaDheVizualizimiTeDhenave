# imporitmi i librarive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import Binarizer
import os

os.chdir("/Users/sedagas/Downloads")

# importimi i datasetit
dataset = pd.read_csv("blockchain_tweets.csv")

# mostrimi
dataset.info()

# definimi i tipeve te dhenave
dataset.dtypes

# Definimi i tipeve te dhenave - permiresimi
# user_created --> tipi object duhet te kthehet ne tipin datetime
# date --> tipi object duhet te kthehet ne tipin datetime
dataset['user_created'] = pd.to_datetime(dataset['user_created'])
dataset['date'] = pd.to_datetime(dataset['date'])
dataset.dtypes

# fshierja kolonave te pa nevojshme
dataset = dataset.drop(['is_retweet'], axis=1)

#transformimi i te dhenave
dataset["user_verified"] = dataset["user_verified"].astype(int)

# strategjia e pastrimit te dhenave
# zevendesimi i vlerave per pastrimin e te dhenave
dataset["user_location"].fillna("No location", inplace=True)
dataset["user_description"].fillna("No description", inplace=True)
dataset["hashtags"].fillna("No hashtags", inplace=True)

# pastirimi i te dhenave duplikate
dataset.drop_duplicates(subset="text")

# aggregimi
dates = dataset.groupby(dataset['date'].dt.day)['date'].agg(['count'])
print('Numri i Blockchain tweets ne baze ditore: \n', dates)

tweets_of_users = dataset.groupby("user_name")["text"].count()
plt.hist(tweets_of_users.head(50))
plt.grid()
plt.show()

# zgjedhja e nenbashkesive --> Numri total i tweeteve te perdouresve
tweets_of_users.head(50)

# diskretizimi
plt.hist(tweets_of_users, bins=10)
plt.grid()
plt.show()

user_followers = pd.cut(dataset['user_followers'], 4)
user_followers = user_followers.apply(lambda x: x.mid)
plt.hist(user_followers)
plt.grid()
plt.show()

# binarizimi i te dhenave
followers = dataset.iloc[:, 4].values
friends = dataset.iloc[:, 5].values

print("\nVlerat origjinale te ndjkesve : \n", followers)
print("\nVlerat origjinale te shokeve : \n", friends)

x = followers
x = x.reshape(1, -1)
y = friends
y = y.reshape(1, -1)

binarizer_for_followers = Binarizer(threshold=1000.0)
binarizer_for_friends = Binarizer(threshold=2000.0)

print("\nNdjekesit e perodruesve ne form te binarizuar : \n", binarizer_for_followers.fit_transform(x))

print("\nShoket e perodruesve ne form te binarizuar : \n", binarizer_for_friends.fit_transform(y))

