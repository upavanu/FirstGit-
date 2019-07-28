import pandas as pd
from surprise import Dataset
from surprise import Reader
# This is the same data that was plotted for similarity earlier
# with one new user "E" who has rated only movie 1
#Load_data
ratings_dict = {
    "item": [1, 2, 1, 2, 1, 2, 1, 2, 1],
    "user": ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E'],
    "rating": [1, 2, 2, 4, 2.5, 4, 4.5, 5, 3],
}
df=pd.DataFrame(ratings_dict)
reader=Reader(rating_scale=(1,5))
#Load Pandas DataFrame
data=Dataset.load_from_df(df[["item","user","rating"]], reader)
#Load builtin movie lens dataset
movielens=Dataset.load_builtin('ml-100k')

#Recommender.py
from surprise import KNNWithMeans
#To use item based cosine similarity
sim_options={
        "name":"cosine",
        "user_based":False #compute similarities between items
        
        }
algo = KNNWithMeans(sim_options=sim_options)