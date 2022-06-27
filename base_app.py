"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
from PIL import Image
#from streamlit_option_menu import option_menu
# Data dependencies
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Customise our plotting settings
sns.set_style('whitegrid')

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options  = ["General Information", "Prediction", "About Us"]
	selected = st.sidebar.selectbox("Navigation", options)
	
	# Building out the "Information" page
	if selected == "General Information":
		image =Image.open ('C:/Users/USER/Pictures/johh/tweeter.jpg')
		st.image(image, use_column_width='always')
		st.write("Climate change is an urgent global issue, with demands for personal, collective, and governmental action. Although a large body of research has investigated the influence of communication on public engagement with climate change, few studies have investigated the role of interpersonal discussion. To continue reading here: [link](https://www.pnas.org/doi/10.1073/pnas.1906589116)")
		st.write("This app is designed to give predictions on the perception or sentiment of the public towards the subject of climate change. Social media being a platform for mass communication is used as a harvest feild to understand how people percieve this to be a problem and shows us how likely solutions are going to be accepted. The platform of reach here is tweeter, and to the top left corner is a drop down to access predictions.\nCheck the box below to see the raw data.")
		#st.subheader("General Information" )
		# You can read a markdown file from supporting resources folder
		#st.markdown("Some information here")
		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
		
		st.write("The above data have been labelled and each tweet divided to any of this four classes based on thier content")
		
		st.write("2 News: the tweet links to factual news about climate change")
		st.write("1 Pro: the tweet supports the belief of man-made climate change")
		st.write("0 Neutral: the tweet neither supports nor refutes the belief of man-made climate change")
		st.write("-1 Anti: the tweet does not believe in man-made climate change")
		
		st.write ("Below is a graphical representation of the distribution of these classes accross the whole data")
		f, ax = plt.subplots(figsize=(8, 4))
		ax = sns.countplot(x="sentiment", data= raw)
		st.pyplot(f)

	# Building out the predication page
	if selected == "Prediction":
		image =Image.open ('C:/Users/USER/Pictures/johh/anime.jpg')
		st.image(image, width = 500)
		#st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Hey you, Let me show you my predictive power. Drop a Tweet in the box below:","Type Here")
		options = ["Logistic Regression", "Mulitnomial Naive Bayes", "Superlearner"]
		Model = st.selectbox("Choose a Model", options, index = 0)
		if Model == "Logistic Regression":
			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				#predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				#prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				#st.success("Text Categorized as: {}".format(prediction))
		elif Model == "Mulitnomial Naive Bayes":
			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				#predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				#prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				#st.success("Text Categorized as: {}".format(prediction))
		elif Model == "Superlearner":
			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				#predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				#prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				#st.success("Text Categorized as: {}".format(prediction))


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
