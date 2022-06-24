#Steps
#1) Base: Build a binary classifier for the 4 targets (4 classifiers) - Patrick
#1a) methodologies to use: different ngrams, different learning algorithms -Chris
#1b) Hyperparameter tuning for: model + optimization algo - Chris
#1c) multiclass models - Patrick
#1d) Optional: tune threshold of classifier towards high-recall/high-precision classifier respectively 

#2) collected metrics for each iteration
#2a) Base: accuracy, precision, recall - Patrick
#2b) Base: make learning curves - Patrick
#2c) Optional: Collect words and ngrams that model learned (for each class)

#3) Analyze models after each iteration
#3a) find strenghts and weaknesses, learning curves, metrcs
#3b) Optional: analyze learned words and ngrams (see 2c) and explain outliers

#4) Logs in a word file for each series of iteration
#4a) save the metrcis of an iteration with the parameters used
#4b) save our comments on the tested models and parameters


#More optional stuff:
#1) try using spacy
#2) use a CNN

import pandas as pd


if __name__ == "__main__":
    train = pd.read_csv("agnews_train.csv")
    test = pd.read_csv("agnews_test.csv")


