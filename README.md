Pneumonia Classifier Demo with covid-19 model.

This project can be launched in Google App Engine Flask App for Pneumonia/CoVID-19 indication in x-ray images, you can clone this repository and deploy to Google Cloud App Engine or any other cloud provider.

The requirements.txt file installs a headless version of opencv for image processing at the prediction stage, this is what was most compatible when deploying to production with Google App Engine, feel free to tinker with it.


To run the classifier in your browser, execute: 
python app.py "-m, --model" optional arguments
