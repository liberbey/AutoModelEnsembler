# AutoModelEnsembler

This is an Automated Model Ensembling framework for classification tasks. With AutoModelEnsembler, you can create ensemble models with many different algorithms and and a meta model. In this initial version, only classification tasks and probability prediction for each class is supported. Also it should be noted that your data should contain only numbers.

Here is an example workflow you can create with AutoModelEnsembler.

![Framework Diagram](https://user-images.githubusercontent.com/37846781/108607311-a4c99c00-73d0-11eb-9184-9464bcecffe6.png)

# How to run?

1) Clone the repository and create a virtual environment.
2) Install necessary libraries provided in requirements.txt.
3) Prepare your configuration file. An example configuration is provided.
4) Start the program with the command: python console/application.py "path/of/config/file"
5) Results will be created as csv files.

# Supported Classifiers

For the first version, only 15 classifiers are supported. If you like to use a different classifier, you should import it in domain/model_controller.py.

* RandomForestClassifier
* AdaBoostClassifier
* GradientBoostingClassifier
* ExtraTreesClassifier
* LogisticRegression
* KNeighborsClassifier
* MLPClassifier
* XGBClassifier
* LinearDiscriminantAnalysis
* QuadraticDiscriminantAnalysis
* LGBMClassifier
* SVC
* GaussianNB
* MultinomialNB
* BernoulliNB

# Future Works

* Support for Regression tasks
* Support for VotingClassifiers
* Usage of multiprocessing for increasing the speed
* Feature Selection
* Support for data with different types
* Database connection
* Hyperparameter optimization using GridSearchCV/optuna
