# movie_recommender_system
This application recommends movies based on similarities between the movie features. This app used GPT-ada-002 for training over 3k embedded dataset.

This movie recommender application, recommends movies with similar attributes. These movies are already available in the dataset. The trained model does not recommend a movie outside of dataset. The objective is to train a custom model for a specific application. This model is an example and can be implemented for other products and services such as:
-	Recommending similar products based on similar features in a retail store.
-	Recommending similar books based on genre and book objective.
-	Recommending similar music.
-	Recommending similar news and articles.
It is important to note that custom models perform better in terms of accuracy, fine tuning, safety, responsible AI guidelines, cost-efficient and quick updates.
We choose GPT text-embedding-ada-002 for training. Embedding is the process of assigning number value to each token. The model first converts the text into tokens, and then assigns vector value to each token in a multi-dimensional space. A cosign function measures the distance between these values. This is how the model understands to group token with similar semantic meanings together, and therefore, makes the prediction or recommendation.
For this project, the training file (dataset) is an excel sheet of columns: Movie_ID, and Movie.
## Steps to train the model:
## 1.	I downloaded the dataset of roughly 3k from Kaggle.
## 2.	Save the dataset as csv-UTF-8.
## 3.	Create a folder in Google Colab and upload the file in the folder. Create a new Colab notebook.
## 4.	First we install the required libraries:
   
! pip install pandas

! pip install openai==0.28.1

! pip install gradio


## 5. import the required libraries
   
import pandas as pd
import openai
import numpy as np
import gradio as gr
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from openai.embeddings_utils import get_embedding

## 6. enter your openai keys:
openai.api_key = ‘openai-api-key'

## 7. load your data.
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/embedding-movie-recommender/large_movie_data.csv')

## 8. We need to store the embedded data in a column. So, we create column “Embedding”.

df['Embedding'] = df['Movie'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
with open('/content/drive/MyDrive/Colab Notebooks/embedding-movie-recommender/large_movie_data_embedded.pkl', 'wb') as f:
  pickle.dump(df, f)
  
## 9. After training, we can check the file, but to see it, we need to convert pickle to csv file.

df = pd.read_pickle('/content/drive/MyDrive/Colab Notebooks/embedding-movie-recommender/large_movie_data_embedded.pkl')
df.to_csv('/content/drive/MyDrive/Colab Notebooks/embedding-movie-recommender/large_movie_data_embedded.csv')

## 10. To test our trained model, we need to load it.
    
with open(‘file_path.pkl) as f:
df = pickle.load(f)

## 11. We create a search function that converts uers input under ‘movie_title’, converts it into embedded value. Then it picks second most similar movie (the first one is the searched movie itself).

def search_movies(df, movie_title, n=2): 
  ### embedding module creates embedding of movie_title (user input) using the same engine.
  embedding = get_embedding(movie_title, engine='text-embedding-ada-002') 
  ### df creates 'similarities' column where it stores similarity value of 'embedding' or (user input movie title) and 'Embedding' embedded values.
  df['similarities']= df.Embedding.apply(lambda x: cosine_similarity([x], [embedding])) 
  ### store the results in a df called res. res contains the top n values in descending order.
  res = df.sort_values('similarities', ascending=False).head(n)
  ### return the searched Movie title and its corresponding similarity value.
  return res.iloc[1]['Movie'], res.iloc[1]['similaritis']

## 12. Finally, we use gradio interface to test our model.
### Define Gradio interface for the recommendation system

def gradio_wrapper(movie_title):
    top_movie, similarity_score = search_movies(df, movie_title)
    return top_movie, similarity_score

iface = gr.Interface(
    fn=gradio_wrapper,
    inputs="text",
    outputs=["text", "number"],
    #interpretation="default",
)
iface.launch(share=True)

