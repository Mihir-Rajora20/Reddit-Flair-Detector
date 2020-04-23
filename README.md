# Reddit-Flair-Detector
A machine learning classifier to predict the flairs of posts of the r/india subreddit deployed to Heroku using the Flask API

# Website 
https://lit-brook-22563.herokuapp.com/
# Index 
1.	Project Description
2.	Data Extraction 
3.	Data Pre-processing and Modification
4.	Model development and Testing
5.	Model Summary
 The **‘Main_Project’** folder contains all the html files and the Python files to create the Flask webapp. 

#Project Description
 The project aims to develop a flair detector for the r/India subreddit. The project was developed in Python in Jupyter Notebooks and     deployed using Flask on Heroku. 
 The Jupyter notebooks and their description are as follows:

### Jupyter Notebooks:
*	Praw Data Extraction :Extracting Data using the Praw API 
*	Data Extraction using pushshifts : Extending the training set using the Pushshifts API
*	Data Pre-Processing: Processing the data in the different datasets and subsequent subsets produced. 
*	Data Exploration and Modification: Modifying the training set, finding data correlation and structure of data values and creating other modified databases. 
*	Model Training: training the different models on variety of features and choosing the final model. 
*	Flair Prediction Model: contains the final model and data processing functions to predict post flair from the input URL. 

### Libraries Used:
The following libraries were used for data exploration and model development –
* Numpy, Pandas 
* Sklearn: for models and feature selection
* Praw for data extraction
* Nltk: for NLP
* Seaborn, matplotlib: for Data visualisation
* GridSearchCV was used for feature selection for the ML models. 

# Data Extraction

 Data was initially extracted using the PRAW module for Reddit data selection from the r/india subreddit.
 Data was sorted on the basis of post scores to select the relatively popular posts which have more text content . 
 The following flairs are present in the r/india subreddit :
 Politics			    Non-Political			    AskIndia	    Policy/Economy	
 Business/Finance 	Science/Technology		Sports		    Food
 Photography 		  CAA-NRC-NPR			      Coronavirus	  Rediquette
 The following number of datapoints were extracted from each flair
<img src="readme_images/praw%20dataset%20post%20count.png" width="800">

**Body was the primary content missing even in the most popular posts** and subsequent post extractions, 
 hence it was removed as a training feature. 

<img src="readme_images/post%20body%20heatmap.png" width="500">

### Using Pushshifts API to get more Data
 To increase the size of our training data, we used the Pushshifts API to extract more posts from Reddit.
 To avoid data duplicacy between this data and the previous extraction, posts dated before 2018 December were removed from the PRAW dataset. 
 
 The following datapoints were obtained from the Pushshifts API
 
<img src="readme_images/pushshifts%20api%20post%20count.png" width="500">

 The 2 datasets were combined to form the **combined_df.csv**. 
 Posts dated before Dec 2018 in the PRAW dataset were removed to avoid  post  duplicacy. 
  
  **Combined_df dataset**
<img src="readme_images/combined%20df%20post%20count.png" width="400">

# Data Pre-Processing and Modification
### Numerical Data Analysis
 No strong correlation was found in the numerical data of posts. Numerical data points like post scores , title lengths , size of comments etc had no correlation to any particular flair and hence were not used for training the model. 

**Post Scores**
<img src="readme_images/Post%20scores.png" width="800">
 Many flairs had high score averages. Scores were not used for training the data.
**Number of Post Comments**
<img src="readme_images/number%20of%20comments.png" width="800">
 
 ### Data Cleaning
 The text data was processed to be used as training data for the ML model. First, posts with null values were removed.
 
                
  <img src="readme_images/null%20values.png">
  
  The title and comment texts for all the posts were cleaned and processed. 
1.	First, text was converted to lower form and all forms of punctuation were removed 
2.	**Word stemming was done using the LancasterStemmer** from nltk.stem and stopwords were removed from the text data, since they don’t contribute and unique identifying words to the text. 
Stemming was chosen since we do not require the actual meaning of the text data, but rather we are searching for unique text identifiers from each flair and do not require a vocabulary to maintain word meaning. 
3.	URL and post titles were processed similarly. 

As is apparent from the flair count for the combined_df dataset, there is a post count disparity between the flairs, with some having exceedingly high number of posts while others having low number of posts. Hence, datasets strip_data and sec_strip_data were created from the original dataset which had an equivalent number of posts for each flair. 
This was done by removing the relatively unpopular posts from the flairs with more post count.

### Extending the list of Stopwords 
To increase model efficiciency, a word count was obtained from each flair using the CountVectorizer and the commonly occurring words from each flair were removed by extending them into the list of stopwords. 



