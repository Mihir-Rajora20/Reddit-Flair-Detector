import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
import praw
nltk.download("stopwords",quiet=True)
stop_words=stopwords.words('english')
lancaster=LancasterStemmer()
import pickle
import warnings
warnings.filterwarnings("ignore")


# GET REDDIT INSTANCE
reddit=praw.Reddit(client_id='Xizrpkw0yLJdBQ',client_secret='IPVu1oXIDtd2jjX8S4PJ499E6vk',user_agent='my_reddit_scraper')
subreddit=reddit.subreddit('india')

### DATA EXTRACTION FROM PRAW

def get_data(post_id_str,topics_dict):
	post_id_list=post_id_str.split()
	for post_id in post_id_list:
	    submission=reddit.submission(id=post_id)
	    topics_dict["flair"].append(submission.link_flair_text)
	    topics_dict["title"].append(submission.title)
	    topics_dict["id"].append(submission.id)
	    topics_dict["url"].append(submission.url)
	    submission.comments.replace_more(limit=0)
	    comment = ''
	    numcomments=0
	    for top_level_comment in submission.comments:
	        comment = comment + ' ' + top_level_comment.body
	        numcomments+=1
	        if numcomments>60:
	            break
	    if(len(comment)==0):
	        topics_dict['comments'].append("No comments")
	    else:
	        topics_dict["comments"].append(comment)
	return topics_dict

#DATA PREPROCESSING FUNCTIONS

def text_process(words): 
    words=str(words)
    if(len(words)==0):
        return [lancaster.stem(word) for word in "No Body"]
    words= BeautifulSoup(words, "lxml").text
    words=words.lower()
    nopunc=[char for char in words if char not in string.punctuation]
    nopunc=''.join(nopunc)
    wordlist= [lancaster.stem(word) for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    wordlist=' '.join(wordlist)
    return wordlist

#url processing function
def url_process(words):
    words=str(words)
    if(len(words)==0):
        return [word for word in "No Body"]
    words= BeautifulSoup(words, "lxml").text
    words=words.lower()
    nopunc=[]
    for char in words:
        if char in string.punctuation:
            nopunc.append(" ")
        else:
            if(char.isdigit()==False):
                nopunc.append(char)
    nopunc=''.join(nopunc)
    stop_words=stopwords.words('english')
    stop_words.append('comments')
    stop_words.append('reddit')
    stop_words.append('https')
    stop_words.append('r')
    stop_words.append('www')
    stop_words.extend(['comments','reddit','https','r','www','com','india','http','html','news'])
    wordlist= [word for word in nopunc.split() if word.lower() not in stop_words]
    wordlist=' '.join(wordlist)
    return wordlist

def text_process2(words):
    words=str(words)
    nopunc=[]
    for char in words:
        if char in string.punctuation:
            nopunc.append(" ")
        else:
            if(char.isdigit()==False):
                nopunc.append(char)
    nopunc=''.join(nopunc)
    stop_words=stopwords.words('english')
    extended_stopwords=['ind','lik','peopl','ev','com','govern','think','bank','tim','work','dont','fuck','new','mak','said','year','nee','want','country','day','giv','thing','good','say','tak','ind','india','peopl']
    stop_words.extend(extended_stopwords)
    wordlist= [word for word in nopunc.split() if word.lower() not in stop_words]
    wordlist=' '.join(wordlist)
    return wordlist

#Check url validity
def check_valid_url(url):
	if "https://www.reddit.com/r/india/comments/" in url:
		s = reddit.submission(url=str(url))

		if s.author is None:
			if s.selftext == '[deleted]':
				return False, "Post deleted"

			if s.selftext == '[removed]':
				return False, "Post has been removed and account has been deleted"

			return False, "Account has been deleted"

		if s.selftext == '[removed]':
			return False, "Post has been Removed"

		return True, "Good to go"

	else:
		return False, "URL either invalid or does not belong to r/india"

#Loading the model
#Getting the prediction

def get_flair_data(post_id):
	topics_dict={"flair":[], "title":[],  "id":[],"url":[], "comments":[]}
	data_dict=get_data(post_id,topics_dict)
	topics_data=pd.DataFrame(data_dict)
	topics_data['stem_comments']=topics_data['comments'].apply(lambda x: text_process(x))
	topics_data['stemmed_titles']=topics_data['title'].apply(lambda x: text_process(x))
	topics_data['stemmed_url']=topics_data['url'].apply(lambda x: url_process(x))
	topics_data['title_comments_stem']=topics_data['stemmed_titles']+topics_data['stem_comments']
	return topics_data





