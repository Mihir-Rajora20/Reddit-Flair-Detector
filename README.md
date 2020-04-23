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
The project aims to develop a flair detector for the r/India subreddit. The project was developed in Python in Jupyter Notebooks and deployed using Flask on Heroku. 
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
![Praw Dataset](readme_images/praw%20dataset%20post%20count.png)

**Body was the primary content missing even in the most popular posts** and subsequent post extractions, 
hence it was removed as a training feature. 

![Missing Features](readme_images/post%20body%20heatmap.png)

### Using Pushshifts API to get more Data
To increase the size of our training data, we used the Pushshifts API to extract more posts from Reddit.
To avoid data duplicacy between this data and the previous extraction, posts dated before 2018 December were removed from the PRAW dataset. 
The following datapoints were obtained from the Pushshifts API
![Pushshifts Post count](readme_images/pushshifts%20api%20post%20count.png)

