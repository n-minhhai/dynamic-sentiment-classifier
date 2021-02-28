# How to Run

> Run "streamlit run app.py" to launch the app in a browser

Within the app, insert the tweet you'd like to perform sentiment classification on and enter the year range of the tweets

# Files
classifier.ipynb - jupyter notebook used for building a sentiment classifier

twitter.ipynb - jupyter notebook used for webscraping, later converted to a python script (app.py) to run as an app

app.py - python application with dynamic sentiment classification of tweets based on user input

# Dataset 

## Dataset for training a sentiment classifier
Amazon app review dataset obtained from: http://jmcauley.ucsd.edu/data/amazon/

## Dataset to perform sentiment classification on
Tweets obtained from Twitter using webscraping (TWINT)

# Classifier
LogisticRegression with 84% accuracy

# Dependencies
- Streamlit
- Twint
- Pandas
- NumPy
- Sklearn
- NLTK
- Asyncio
- nest_asyncio