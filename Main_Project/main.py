from flask import Flask, render_template,request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import pickle
import script
import os
import sklearn
app = Flask(__name__)


@app.route("/flair", methods=['POST'])
def flair():
	URL = request.form['redditURL']

	check_error, error_message = script.check_valid_url(URL)

	if check_error == False:
		return render_template('error_gen_page.html', error_message=error_message)

	else:
		post_id=script.get_id(str(URL))
		prediction,actual_flair=script.predict(post_id)
		dirname = os.path.dirname(__file__)

		filename = str(dirname) + "trained_pipeline_pickle.sav"
		print(dirname, " ",filename)
		predicted_flair = prediction
		return render_template('flair_output_page.html', predicted_flair=prediction,actual_flair=actual_flair)
                                                

@app.route("/automated_testing", methods=['POST'])
def automated_testing_file():
	if 'attachment' not in request.files:
		if 'url' not in request.args:
			return render_template('error_gen_page.html', error_message="No input provided")

		else:
			URL = request.args['url']
			check_error, error_message = script.check_valid_url(URL)

			if check_error == False:
				return error_message

			else:
				post_id=script.get_id(str(URL))
				prediction,actual_flair=script.predict(post_id)
				dirname = os.path.dirname(__file__)
				filename = str(dirname) + "trained_pipeline_pickle.sav"
				print(dirname, " ",filename)
				
				predicted_flair = prediction

				final_result = {
					"key": URL,
					"value": predicted_flair
				}

				return jsonify(final_result)

	elif 'attachment' in request.files:
		dirname = os.path.dirname(__file__)
		filename = str(dirname) + "trained_pipeline_pickle.sav"
		print(dirname, " ",filename)
		loaded_model = pickle.load(open(filename, 'rb')) # removed in app.py
		attachment = request.files['attachment']

		urls = attachment.read()
		urls = urls.decode("utf-8")

		print("\n\n")
		print(urls)
		print("\n\n")

		urls_list = str(urls).split('\n')

		print(urls_list)

		array = []

		for url in urls_list:
			check_error, error_message = script.check_valid_url(url)

			if check_error == False:
				element = {
					"key": url,
					"value": str("Link error " + error_message)
				}
				array.append(element)

			else:

				post_id=script.get_id(str(url))
				prediction,actual_flair=script.predict(post_id)

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

