import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

data = pd.read_csv('dataset.csv')

data = data[['id', 'title', 'overview', 'genre']]
data['tags'] = data['overview']+data['genre']
new_data = data.drop(columns=['overview', 'genre'])

cv = CountVectorizer(max_features=10000, stop_words='english')
vector = cv.fit_transform(new_data['tags'].values.astype('U')).toarray()

similarity = cosine_similarity(vector)

def recommend(movies):
    index = new_data[new_data['title']==movies].index[0]
    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector:vector[1])
    for i in distance[0:5]:
        print(new_data.iloc[i[0]].title)

recommend('Iron Man')
pickle.dump(new_data, open('movies_list.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
pickle.load(open('movies_list.pkl', 'rb'))
