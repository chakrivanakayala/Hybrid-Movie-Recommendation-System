#import Flask 
from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd
import requests
from dotenv import load_dotenv, find_dotenv
import os
load_dotenv("/scr/.env")


MoviesData= joblib.load('Movies_Datase.pkl') 
movies = MoviesData['title']
X = joblib.load('Movies_Learned_Features.pkl') 
my_ratings = np.zeros((9724,1))
my_movies=[]
my_added_movies=[]

def computeCost(X, y, theta):
   m=y.size
   s=np.dot(X,theta)-y
   j=(1/(2*m))*(np.dot(np.transpose(s),s))
   return j

def gradientDescent(X, y, theta, alpha, num_iters):  
    m = float(y.shape[0])
    theta = theta.copy()
    for i in range(num_iters):
        theta=(theta)-(alpha/m)*(np.dot(np.transpose((np.dot(X,theta)-y)),X))
    return theta

#Adding movies and ratings given by user
def checkAndAdd(movie,rating):
        movie = str(movie).lower()
        index=MoviesData[MoviesData['title']==movie].index.values[0]
        my_ratings[index] = rating
        movieid=MoviesData.loc[MoviesData['title']==movie, 'movieId']
        if movie in my_added_movies:
                return (1)
        my_movies.append(movieid)
        my_added_movies.append(movie)
        return (0)

#creating an instance of Flask
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html', movies = movies)

@app.route('/addMovie/', methods=['GET','POST'])
def addMovie():
    val=request.form.get('movie')
    rating=request.form.get('rating')
    flag=checkAndAdd(val,rating)
    if (flag==1):
            processed_text="The movie has already been added by you"
            return render_template('home.html',processed_text=processed_text, movies = movies)
    else:        
            processed_text="Successfully added movie to your rated movies"
            movie_text=", you've rated "+rating+" stars to movie: "+val
            return render_template('home.html',processed_text=processed_text,movie_text=movie_text,my_added_movies=my_added_movies, movies = movies)
    
@app.route('/reset/', methods=['GET','POST'])
def reset():
        global my_ratings
        global my_movies
        global my_added_movies
        
        my_ratings = np.zeros((9724,1))
        my_movies=[]
        my_added_movies=[]
        processed_text='Successfull reset'
        return render_template('home.html',processed_text=processed_text, movies = movies)

@app.route('/predict/', methods=['GET','POST'])
def predict(flag=None):
    if request.method == "POST":
        if (len(my_added_movies)==0):
                processed_text="Yikes! you've to add some movies before predicting anything "
                return render_template('home.html',processed_text=processed_text, movies = movies)
        if(flag==1):
                if (len(my_added_movies)==0):
                        processed_text="Yikes! you've to add some movies before predicting anything "
                        return render_template('home.html',processed_text=processed_text, movies = movies)
                        #get form data
                out_arr = my_ratings[np.nonzero(my_ratings)]
                out_arr=out_arr.reshape(-1,1)
                idx = np.where(my_ratings)[0]
                X_1=[X[x] for x in idx]
                X_1=np.array(X_1)
                y=out_arr
                y=np.reshape(y, -1)
                theta = gradientDescent(X_1,y,np.zeros((100)),0.001,4000)
                
                p = X @ theta.T
                p=np.reshape(p, -1)
                predictedData=MoviesData.copy()
                predictedData['Pridiction']=p
                sorted_data=predictedData.sort_values(by=['Pridiction'],ascending=False)
                sorted_data=sorted_data[~sorted_data.title.isin(my_added_movies)]
                sorted_data=sorted_data[:40]
                return(sorted_data)
                
        
        #get form data
        out_arr = my_ratings[np.nonzero(my_ratings)]
        out_arr=out_arr.reshape(-1,1)
        idx = np.where(my_ratings)[0]
        X_1=[X[x] for x in idx]
        X_1=np.array(X_1)
        y=out_arr
        y=np.reshape(y, -1)
        theta = gradientDescent(X_1,y,np.zeros((100)),0.001,4000)
        
        p = X @ theta.T
        p=np.reshape(p, -1)
        predictedData=MoviesData.copy()
        predictedData['Pridiction']=p
        sorted_data=predictedData.sort_values(by=['Pridiction'],ascending=False)
        sorted_data=sorted_data[~sorted_data.title.isin(my_added_movies)]
        sorted_data=sorted_data[:40]
        my_list=sorted_data.values.tolist()
        titles = [title[1].strip() for title in my_list]
        list = []
        api = os.getenv("API_KEY")
        for title in titles:
              response = requests.get(f"http://www.omdbapi.com/?t={title}&apikey={api}").json()
              if(response['Response'] == 'True'):
                        list.append(response)
        return render_template('result.html',list=list)
    #pass

if __name__ == '__main__':
    app.run()
