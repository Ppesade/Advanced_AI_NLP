#Steps
#1) Base: Build a binary classifier for the 4 targets (4 classifiers) - Patrick
#1a) methodologies to use: different ngrams, different learning algorithms -Chris 
#1b) Hyperparameter tuning for: model + optimization algo - Chris
#1c) multiclass models - Patrick ; Comment: convenience functions we wrote to inspect weights assume a binary weight vector
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

#Imports

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, learning_curve

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

#Reporting
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Models used

#1) naive bayes model
mr_naivebayes = MultinomialNB()

#2) SGD Classifier
sgd =  SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)

#3) SGD Classifier multiclass
sgd_multiclass =  SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)



#Pipeline

def define_pipeline(model):
    pipeline = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('model', model),
                ])
    return pipeline

# Link for SGD: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
# Link for Naive Bayes: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html

def define_param_grid(ngram, alpha):
    """
    Creates a dictionary of the range of ngram values and alpha values you want to take in consideration.
    Params:
    - ngram: int. Returns a list of ranges of ngram in tuples: (1, 1) up to (1, ngram)
    - alpha: int. Returns a list of ranges of alpha in tuples: 0.001 to 10^alpha in steps of 10^-3
    """
    #ngram range for hyperparameter tuning
    vect__ngram_range = []
    for i in range(ngram):
        vect__ngram_range.append((1,i+1))
    #alpha range for hyperparameter tuning
    model__alpha = [10 ** - (x*3) for x in range(1, alpha + 1)]
    #create dictionary of model grid parameters
    model_grid_parameters = {
    'vect__ngram_range': vect__ngram_range,
    'model__alpha': model__alpha
}
    return model_grid_parameters
    

def define_gridsearch(model: object, param_grid: dict, scorer = "accuracy"):
    pipe = define_pipeline(model)
    grid = GridSearchCV(pipe, param_grid, cv = 5, scoring = scorer, return_train_score = True)
    return grid

def crossvalidation_report_df(grid_cv):

    """Convenience function.
    Creates a simple dataframe that reports the results of a
    cros-validation experiment. The input grid_cv must be fit.
    Returns a dataframe, sorted by rank of experiment.
    """
    # pick columns that define each experiment (start with param)
    # and the columns that report mean_test and rank_test results
    cols = [c for c in grid_cv.cv_results_ if (c.startswith('param') or c in ['mean_test_score', 'rank_test_score'])]

    # sort original df by rank, and select columns
    return pd.DataFrame(grid_cv.cv_results_).sort_values(by='rank_test_score')[cols]

def sort_feature_weights(grid, fkey='vect', wkey='model'):
    ''' Convenience function.
    Gets the weights of each words/ngram and orders them from lowest to highest. 
    Highest being words most associated with the given topic
    Returns a list of tuples with word/ngram and its weight'''
    F = grid.best_estimator_[fkey].get_feature_names_out()
    W = grid.best_estimator_[wkey].coef_[0]
    return sorted(zip(F, W), key=lambda fw: fw[1]) 

def sort_feature_multiclassweights(grid, fkey='vect', wkey='model'):
    ''' Convenience function.
    Gets the weights of each words/ngram and orders them from lowest to highest. 
    Highest being words most associated with the given topic
    Returns a list of tuples with word/ngram and its weight'''
    F = grid.best_estimator_[fkey].get_feature_names_out()
    science = grid.best_estimator_[wkey].coef_[0]
    sports = grid.best_estimator_[wkey].coef_[1]
    world = grid.best_estimator_[wkey].coef_[2]
    business = grid.best_estimator_[wkey].coef_[3]

    return sorted(zip(F, science, sports, world, business), key=lambda fw: fw[1]) 


def apply_modelling(grid, train_data, test_data):
    ''' Convenience function.
    Creates the model for all news types and returns all the info we need to analyze the model
    outputs:
    cv_report: df report of the CV with the params tried and their scores in the CV
    score_report: report of the best_estimator's performance on the test data
    bestestimator_parameters: parameters used in the best estimator
    bestestimator_weights: word weigths for the best estimator
    '''
    
    targets = ["science_int", "sports_int", "world_int", "business_int"]
    report_dict = {
        "targets": targets, 
        "cv_report":[], 
        "score_report":[],
        "bestestimator_parameters":[],
        "grid":[], 
        "bestestimator_weights":[]
    }

    for target in targets:
        grid.fit(train_data.text, train_data.loc[:,target])
        cv_report = crossvalidation_report_df(grid)
        score_report = classification_report(test_data.loc[:,target], 
            grid.best_estimator_.predict(test_data.text),target_names = ["others", target[:-4]])
        bestestimator_parameters = grid.best_params_
        bestestimator_weights = sort_feature_weights(grid) 
        report_dict["cv_report"].append(cv_report)
        report_dict["score_report"].append(score_report)
        report_dict["bestestimator_parameters"].append(bestestimator_parameters)
        report_dict["grid"].append(grid)
        report_dict["bestestimator_weights"].append(bestestimator_weights)
    return report_dict


def apply_multiclassmodelling(grid, train_data, test_data):
    ''' Convenience function.
    Creates the multiclass-model and returns all the info we need to analyze the model
    outputs:
    cv_report: df report of the CV with the params tried and their scores in the CV
    score_report: report of the best_estimator's performance on the test data
    bestestimator_parameters: parameters used in the best estimator
    bestestimator_weights: word weigths for the best estimator
    '''
    
    targets = ["label_int"]
    report_dict = {
        "targets": targets, 
        "cv_report":[], 
        "score_report":[],
        "bestestimator_parameters":[],
        "grid":[], 
        "bestestimator_weights":[]
    }

    for target in targets:
        grid.fit(train_data.text, train_data.loc[:,target])
        cv_report = crossvalidation_report_df(grid)
        score_report = classification_report(
            test_data.loc[:,target], 
            grid.best_estimator_.predict(test_data.text),
            target_names = ["Science", "Sports", "World", "Business"])
        bestestimator_parameters = grid.best_params_
        bestestimator_weights = sort_feature_multiclassweights(grid) #order of weights is like in tabl (science first,...)
        report_dict["cv_report"].append(cv_report)
        report_dict["score_report"].append(score_report)
        report_dict["bestestimator_parameters"].append(bestestimator_parameters)
        report_dict["grid"].append(grid)
        report_dict["bestestimator_weights"].append(bestestimator_weights)
    return report_dict


def create_learningcurve(train_data, test_data, selected_estimator, loss, report_dict):
    ''' Create the learning curve for the selected estimator
    returns the report_dict with an added column for the learning curve'''
    targets = ["science_int", "sports_int", "world_int", "business_int"]
    report_dict["training_curve"] = []
    for target in targets:
        
        #Takes a loss function and a model and find the performance for a given amount of data for a specific data set
        training_curve = np.array([[0,0]])
        n_samples = 10
        while n_samples < train_data.shape[0]:
            print (n_samples)
            i=0
            new_model = selected_estimator.fit(
                train_data.loc[:n_samples,"text"], train_data.loc[:n_samples,target]
                )
            y_pred = new_model.predict(test_data.text)
            score = loss(test_data.loc[:,target], y_pred)
            training_curve = np.append(training_curve,[[n_samples, score]], axis = 0)
            n_samples *= 2
        report_dict["training_curve"].append(training_curve)
    return report_dict

def create_multiclasslearningcurve(train_data, test_data, selected_estimator, loss, report_dict):
    ''' Create the multiclass learning curve for the selectede estimator
    returns the report_dict with an added column for the learning curve'''
    targets = ["label_int"]
    report_dict["training_curve"] = []
    for target in targets:
        
        #Takes a loss function and a model and find the performance for a given amount of data for a specific data set
        training_curve = np.array([[0,0]])
        n_samples = 10
        while n_samples < train_data.shape[0]:
            print (n_samples)
            i=0
            new_model = selected_estimator.fit(
                train_data.loc[:n_samples,"text"], train_data.loc[:n_samples,target]
                )
            y_pred = new_model.predict(test_data.text)
            score = loss(test_data.loc[:,target], y_pred)
            training_curve = np.append(training_curve,[[n_samples, score]], axis = 0)
            n_samples *= 2
        report_dict["training_curve"].append(training_curve)
    return report_dict


##Analysis functions that can be used to dig into the models##

def show_cv_report():
    ''' Convenience function to print results to analyze them'''
    targets = ["science_int", "sports_int", "world_int", "business_int"]
    for index, value in enumerate(targets):
        print("\n CV report for {}\n".format(value[:-4]))
        print(report_dict["cv_report"][index])

def show_classification_report():
    ''' Convenience function to print results to analyze them'''
    targets = ["science_int", "sports_int", "world_int", "business_int"]
    for index, value in enumerate(targets):
        print("\n Classification report of best estimator for {}\n".format(value[:-4]))
        print(report_dict["score_report"][index])

def show_best_estimator():
    ''' Convenience function to print results to analyze them'''
    targets = ["science_int", "sports_int", "world_int", "business_int"]
    for index, value in enumerate(targets):
        print("\n Parameters of best estimator for {}\n".format(value[:-4]))
        print(report_dict["bestestimator_parameters"][index])    

def show_word_weights():
    ''' Convenience function to print results to analyze them'''
    targets = ["science_int", "sports_int", "world_int", "business_int"]
    for index, value in enumerate(targets):
        print("\n Top words for {} \n".format(value[:-4]))
        print(report_dict["bestestimator_weights"][index][-20:])
        print("\n Worst words for {}\n".format(value[:-4]))
        print(report_dict["bestestimator_weights"][index][:20])      

def show_multiclassword_weights():
    ''' Convenience function to print results to analyze them'''
    targets = ["science_int", "sports_int", "world_int", "business_int"]
    for index, value in enumerate(targets):
        sorted_words = sorted(multiclass_report_dict["bestestimator_weights"][0], key=lambda fw: fw[index + 1])
        print("\n Top words for {}\n".format(value[:-4]))
        print(sorted_words[-20:])
        print("\n Worst words for {}\n".format(value[:-4]))
        print(sorted_words[:20])      


def show_learning_curve():
    ''' Convenience function to print results to analyze them'''
    targets = ["science_int", "sports_int", "world_int", "business_int"]
    for index, value in enumerate(targets):
        print("\n Learning curve for {}\n".format(value[:-4]))
        x = report_dict["training_curve"][index][:,0]
        y = report_dict["training_curve"][index][:,1]
        plt.plot(x,y, label = value[:-4])
    plt.ylabel("Accuracy")
    plt.xlabel("# of samples in training")
    plt.legend()
    plt.show()

def show_multiclasslearning_curve():
    ''' Convenience function to print results to analyze them'''
    targets = ["label_int"]
    for index, value in enumerate(targets):
        print("\n Learning curve for {}\n".format(value))
        x = multiclass_report_dict["training_curve"][index][:,0]
        y = multiclass_report_dict["training_curve"][index][:,1]
        plt.plot(x,y, label = "Overall")
    plt.ylabel("Accuracy")
    plt.xlabel("# of samples in training")
    plt.legend()
    plt.show()


# n-gram instead of bag of words to take in consideration word order and not shuffle meaning

def make_sklearn_analyzer(n=1):
    """Creates a text analyzer from sklearn that extracts n-grams."""
    vect = CountVectorizer(stop_words=[], ngram_range=(1,n))
    analyzer = vect.build_analyzer()
    return analyzer




if __name__ == "__main__":

    #Load data
    train = pd.read_csv("./06_Session 6 - NLP/Advanced_AI_NLP/agnews_train.csv")
    test = pd.read_csv("./06_Session 6 - NLP/Advanced_AI_NLP/agnews_test.csv")


    ####### models for each target ######

    #Define model
    param_grid = {
        'vect__ngram_range': [(1,1)],
        'model__alpha': [(1e-9)]
        }    
    grid = define_gridsearch(model = sgd, param_grid = param_grid, scorer = "accuracy")
    
    #Create reporting data
    report_dict = apply_modelling(grid, train, test)
    report_dict = create_learningcurve(train, test, grid.best_estimator_, accuracy_score, report_dict) #can choose any estimator; don't have to choose best_estimator here

    #Saving the results of the testing into 
    df = pd.DataFrame(report_dict) 
    df.to_csv (r"./06_Session 6 - NLP/Advanced_AI_NLP/base_test_series_data.csv",index = False, header=True)

    ##Analysis##

    #looking at the performance of the different hyperparameters in the gridsearch
    print("CV report for each target")
    show_cv_report()

    #Looking at the classification report of the best estimator
    print("Classification reports of best estimators")
    show_classification_report()

    #Looking at the parameters of the best estimator
    print("Parameters of best estimators")
    show_best_estimator()

    #Look at word weights
    print("Word weights of best estimators for each target")
    show_word_weights()


    #looking at the learning curve
    print("Leanring curve of models")
    show_learning_curve()



    ####### Multiclass models ######

    #Define model
    param_grid = {
        'vect__ngram_range': [(1,1)],
        'model__alpha': [(1e-9)]
        }    
    multiclass_grid = define_gridsearch(model = sgd_multiclass, param_grid = param_grid, scorer = "accuracy") #Not sure whether multiclass would work with the bayes

    #Create reporting data
    multiclass_report_dict = apply_multiclassmodelling(multiclass_grid, train, test)
    multiclass_report_dict = create_multiclasslearningcurve(train, test, multiclass_grid.best_estimator_, accuracy_score, multiclass_report_dict) #can choose any estimator; don't have to choose best_estimator here

    #Saving the results of the testing into 
    multiclass_df = pd.DataFrame(multiclass_report_dict) 
    multiclass_df.to_csv (
        r"./06_Session 6 - NLP/Advanced_AI_NLP/multiclass_base_test_series_data.csv",
         index = False, header=True
         )


    ##Analysis##

    #looking at the performance of the different hyperparameters in the gridsearch
    print("\n Gridsearch CV report \n")
    print(multiclass_report_dict["cv_report"][0])      

    #Looking at the classification report of the best estimator
    print("\n Classification report of best estimator \n")
    print(multiclass_report_dict["score_report"][0])

    #Looking at the parameters of the best estimator
    print("\n Parameters of best estimator \n")
    print(multiclass_report_dict["bestestimator_parameters"][0])

    #Look at word weights
    print("\n Word weights for each target \n")
    show_multiclassword_weights()


    #looking at the learning curve
    print("\n Learning curve of models \n")
    show_multiclasslearning_curve()