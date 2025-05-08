You can use the included pipfile to set up the environment:

`pipenv install`

- preprocess_data.py extracts features from wav files, generating the csv files found in the datasets folder.
  For this to function, you must download the gtzan dataset and place it in datasets.
- models.py runs machine learning methods (KNN, SVM, and random forest) on the processed features.
- results.txt displays the results of running models.py.
