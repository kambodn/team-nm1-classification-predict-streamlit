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
import re
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from streamlit_lottie import st_lottie
import string
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
	

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options  = ["General Information", "Prediction", "About Us"]
	selected = st.sidebar.selectbox("Navigation", options)
	
	# Building out the "Information" page
	if selected == "General Information":
		st.title("DataFluent Tweet Classifier")
		st.subheader("Climate change tweet classification")
		image =Image.open ('C:/Users/USER/Pictures/johh/analy 2.jpg')
		st.image(image, use_column_width = True)
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
		st.markdown('#')
		



		def hashtag_extract(tweets):

			"""
			The function takes in tweets as input and extracts the hashtag from the tweets
			using (re) and returns the hashtags
			
			"""
			hashtags=[]
			tweets=tweets.to_list()
			for tweet in tweets:
				hashtag = re.findall(r"#(\w+)",tweet)
				hashtags.append(hashtag)
			return hashtags

		pro_climate = hashtag_extract(raw['message'][raw['sentiment']==1])

		# Extract Hashtags for anti climate change 
		anti_climate = hashtag_extract(raw['message'][raw['sentiment']==-1])
		pro_climate = sum(pro_climate, [])
		anti_climate = sum(anti_climate, [])

		freq = FreqDist(pro_climate)
		freq_df = pd.DataFrame({'hashtags':freq.keys(),
							'counts':freq.values()})

		freq1 = FreqDist(anti_climate)
		freq_df1 = pd.DataFrame({'hashtags':freq1.keys(),
						'counts':freq1.values()})
		# Display the top 10 frequent hashtags
		st.markdown('##')
		st.subheader("The HashTags")
		st.write('Considering the various Hashtags that trended on tweeter regarding climate change, we can have a view of the inclination of people towards being pro-climate and anti-climate change')
		if st.checkbox('Show plot of Pro-Climate hashtags'):
			freq_df = freq_df.nlargest(columns='counts', n=10)
			g = plt.figure(figsize=(15,10))
			sns.barplot(data=freq_df, x='hashtags',y='counts')
			st.pyplot(g)
		if st.checkbox('Show plot of Anti-Climate hashtags'):
			freq_df1 = freq_df1.nlargest(columns='counts', n=10)
			t = plt.figure(figsize=(15,10))
			sns.barplot(data=freq_df1, x='hashtags',y='counts')
			st.pyplot(t)
			st.markdown ('#')
			st.write ('Yes you guessed right')
			image =Image.open ('C:/Users/USER/Pictures/johh/Trump.jpg')
			st.image(image, width = 400)
		
	def remove_punctuation(text):
		string2 = ''.join([l for l in text if l not in string.punctuation])
		return string2



	# Building out the predication page
	if selected == "Prediction":
		st.title("Sentiment Analysis")
		image =Image.open ('C:/Users/USER/Pictures/johh/anime.jpg')
		st.image(image, width = 500)
		#st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Hey you, Let me show you my predictive power. Drop a Tweet in the box below:","Type Here")
		options = ["Logistic Regression", "Mulitnomial Naive Bayes", "Random Forest Classifier", "Linear Support Vector Classifier"]
		Model = st.selectbox("Choose a Model", options, index = 0)
		if Model == "Logistic Regression":
			if st.button("Classify"):
				# Transforming user input with vectorizer
			
				vect_text = tweet_cv.transform([train['message']]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("mlr_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				st.success("This Text is Categorized as: {}".format(prediction))
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
		elif Model == "Random Forest Classifier":
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
		elif Model == "Linear Support Vector Classifier":
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
	
	# Building out the predication page
	if selected == "About Us":
		st.title("DataFluent Inc.")
		st.subheader (" Who are we?")
		st.write ('DataFluent is a Machine learning and A.I solutions service provider bordering on exploiting data to solve climate change related problems for clients all over the globe. Data is the life, and as such we dive into the unseen and unharnessed vastness of its universe to seek answers to problems that threaten our world.')
		col1, col2 = st.columns(2)

		with col1:
			st.subheader("Our Team")
			st.write("Chibuike")
			st.write("David  ")
			st.write("John")
			st.write("Rabe")
			st.write("Endurance")

		with col2:
			st.subheader("Designation")
			st.write(" Team Lead")
			st.write("Tech Lead")
			st.write("App Deployment")
			st.write("Admin")
			st.write("Comet version")



# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
