# Student Query Classification

Classifying student queries with NLP + machine learning.

Dataset: Student Queries Data.xlsx (22094 rows, 21519 after cleaning)
Classes: Academic, Course Query, Technical

Features: tf-idf (unigrams + bigrams, sublinear tf, min_df=2).
LR, SVM and the neural net use all 42643 features, random forest and xgboost use the top 3000.

Models: logistic regression, linear SVM, random forest, xgboost (if installed), neural network (MLP), all tuned, plus an LSTM (only if tensorflow is installed, it isn't saved into the API)

Best model: Tuned Linear SVM
Accuracy 0.6296 | Precision 0.6262 | Recall 0.6296 | F1 0.6255

Run the API with `python app.py` (http://127.0.0.1:5000), then POST to /predict:

{"query": "How do I register for a course?"}
