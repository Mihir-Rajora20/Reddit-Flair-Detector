from flask import Flask, render_template,request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import pickle
import script
import os
import sklearn
import pandas as pd
#from celery import Celery

app = Flask(__name__)



@app.route("/flair", methods=['POST'])
def flair():
	URL = request.form['redditURL']

	check_error, error_message = script.check_valid_url(URL)

	if check_error == False:
		return render_template('error_gen_page.html', error_message=error_message)

	else:

		post_id= (str(URL))[40:46]
		filename='trained_pipeline_pickle.sav'
		model_load=pickle.load(open(filename, 'rb'))

		topics_data=script.get_flair_data(post_id)
		req_data=topics_data['title_comments_stem']+topics_data['stemmed_url']
		pred_flair=model_load.predict(req_data)
		pred_flair=pred_flair[0]
		prediction,actual_flair=pred_flair,topics_data['flair']


		dirname = os.path.dirname(__file__)

		filename = str(dirname) + "trained_pipeline_pickle.sav"
		print(dirname, " ",filename)
		predicted_flair = prediction
		return render_template('flair_output_page.html', predicted_flair=prediction,actual_flair=actual_flair)
                                                

@app.route("/automated_testing", methods=['POST'])
def automated_testing_file():
	
	#to prevent model loading multiple times, we will define predict function here only
	

	if 'upload_file' not in request.files:
		if 'url' not in request.args:
			return render_template('error_gen_page.html', error_message="No input provided")

		else:
			URL = request.args['url']
			check_error, error_message = script.check_valid_url(URL)

			if check_error == False:
				return error_message

			else:

				print(dirname, " ",filename)
				
				predicted_flair = prediction

				final_result = {
					"key": URL,
					"value": predicted_flair
				}

				return jsonify(final_result)

	elif 'upload_file' in request.files:
		dirname = os.path.dirname(__file__)
		filename = str(dirname) + "trained_pipeline_pickle.sav"
		print(dirname, " ",filename)
		upload_file = request.files['upload_file']

		urls = upload_file.read()
		urls = urls.decode("utf-8")

		print("\n\n")
		print(urls)
		print("\n\n")

		urls_list = str(urls).split('\n')

		print(urls_list)

		array = []
		#arr=[]
		filename='trained_pipeline_pickle.sav'
		model_load=pickle.load(open(filename, 'rb'))
		#array=automated_testing_output.delay(urls_list,arr)  try and add this to redis backend


		for url in urls_list:
			check_error, error_message = script.check_valid_url(url)

			if check_error == False:
				element = {
					"key": url,
					"value": str("Link error " + error_message)
				}
				array.append(element)

			else:
				post_id= (str(url))[40:46]

				topics_data=script.get_flair_data(post_id)
				req_data=topics_data['title_comments_stem']+topics_data['stemmed_url']
				pred_flair=model_load.predict(req_data)
				pred_flair=pred_flair[0]
				prediction,actual_flair=pred_flair,topics_data['flair']

				element = {
					"key": url,
					"value": prediction
				}

				array.append(element)
			
		return jsonify(array[:-1])

@app.route("/")
def home():
	return render_template("main_page.html")

if __name__ == "__main__":
	app.run(threaded=True, port=5000)

