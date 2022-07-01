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
news_vectorizer = open("resources/vectorizer.pkl","rb")
tweet_vect = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options  = ["General Information", "Data Exploration and Analysis", "Prediction", "About Us"]
	selected = st.sidebar.selectbox("Navigation", options)
	
	# Building out the "Information" page
	if selected == "General Information":
		st.title("DataFluent Tweet Classifier")
		st.subheader("Climate change tweet classification")
		st.image("http://www.noaa.gov/sites/default/files/styles/landscape_width_1275/public/2022-03/PHOTO-Climate-Collage-Diagonal-Design-NOAA-Communications-NO-NOAA-Logo.jpg")
		st.write("Climate change is an urgent global issue, with demands for personal, collective, and governmental action. Although a large body of research has investigated the influence of communication on public engagement with climate change, few studies have investigated the role of interpersonal discussion. To continue reading here: [link](https://www.pnas.org/doi/10.1073/pnas.1906589116)")
		st.subheader("Sentiments by Country")
		st.write("Senitments around climate change vary from country to country based on widespread information on the subject, the socio-political inclination of the country, Economic development and Literacy levels. The checkbox can be used to see the dominant sentiments accross some countries of the world.")
		st.image ('https://d25d2506sfb94s.cloudfront.net/cumulus_uploads/inlineimage/2019-09-13/Chart%201-US-01.png', width = 600)

		#https://www.worldatlas.com/r/w960-q80/upload/bb/a5/20/shutterstock-271128437.jpg
	if selected == "Data Exploration and Analysis":	
		st.title("Data Exploration and Analysis")
		st.subheader("Taking a look at all tweets and Hashtags")
		st.image('https://www.maxqda.com/wp/wp-content/uploads/sites/2/Blog-Twitter2-1024x538.png')
		st.markdown('#')
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
		st.markdown('#')
		st.write ("Below is a graphical representation of the distribution of these classes accross the whole data")
		f, ax = plt.subplots(figsize=(8, 4))
		ax = sns.countplot(x="sentiment", data= raw)
		st.pyplot(f)


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
		st.markdown('#')
		st.subheader("The HashTags")
		st.write('Considering the various Hashtags that trended on tweeter regarding climate change, we can have a view of the inclination of people towards being pro-climate and anti-climate change')
		st.write('Show plot of Pro-Climate hashtags')
		freq_df = freq_df.nlargest(columns='counts', n=10)
		g = plt.figure(figsize=(10,5))
		sns.barplot(data=freq_df, x='hashtags',y='counts')
		st.pyplot(g)
		st.markdown('#')
		st.write('Show plot of Anti-Climate hashtags')
		freq_df1 = freq_df1.nlargest(columns='counts', n=10)
		t = plt.figure(figsize=(10,5))
		sns.barplot(data=freq_df1, x='hashtags',y='counts')
		st.pyplot(t)
		#image =Image.open ('')
		if st.checkbox ("Can you check this out?"):
			st.image("https://www.toonpool.com/user/27740/files/trump_and_the_climate_change_2986815.jpg")
			st.write( 'Yep, you know who')
	def remove_punctuation(text):
		string2 = ''.join([l for l in text if l not in string.punctuation])
		return string2



	# Building out the predication page
	if selected == "Prediction":
		st.title("Sentiment Analysis")
		#image =Image.open ('C:/Users/USER/Pictures/johh/anime.jpg')
		st.image("https://netbasequid.com/wp-content/uploads/Social-Sentiment-Analysis.jpg")
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Lets see how this model works:","Type Here")
		
		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_vect.transform([tweet_text])
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("log_reg_model.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			output = {
				2: 'News',
				1: 'Pro-climate change',
				0: 'Neutral',
				-1: 'Anti-climate change'
			}
			st.success(f"This Text is Categorized as: {output[int(prediction)]}")

	# Building out the predication page
	if selected == "About Us":
		st.title("DataFluent Inc.")
		st.subheader (" Who are we?")
		st.write ('DataFluent is a Machine learning and A.I solutions service provider bordering on exploiting data to solve climate change related problems for clients all over the globe. Data is the life, and as such we dive into the unseen and unharnessed vastness of its universe to seek answers to problems that threaten our world.')
		st.subheader("Meet Our Team")
		col1, col2, col3 = st.columns(3)

		with col1:
			st.image("https://cdn.discordapp.com/attachments/986578789431672892/991953630020255764/gt3.jpg", width = 200)
			st.write ('Godwin, Team Lead')
		with col2:
			st.image("https://media.discordapp.net/attachments/986578789431672892/991953475749548164/kambo.jpg", width = 200)
			st.write ('David, Tech Lead')
		with col3:
			st.image("https://cdn.discordapp.com/attachments/986578789431672892/992184796753186926/WhatsApp_Image_2022-06-30_at_8.48.54_PM_1.jpeg", width = 180)
			st.write ('Rabe, Admin Lead')

		col1, col2, col3 = st.columns(3)

		with col1:
			st.image("https://cdn.discordapp.com/attachments/986578789431672892/992184796426027100/WhatsApp_Image_2022-06-30_at_8.48.54_PM_2.jpeg", width = 200)
			st.write ('John, App Dev.')
		with col2:
			st.image("https://cdn.discordapp.com/attachments/986578789431672892/991953374092206180/Peter.jpg", width = 200)
			st.write ('Peter, Sr. Data Scientist')
		with col3:
			st.image("https://cdn.discordapp.com/attachments/986578789431672892/992187396034666586/WhatsApp_Image_2022-06-30_at_8.48.53_PM.jpeg", width = 180)
			st.write ('Endurance, Sr. Data Scientist')

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
