# Description for "mantle_source_classifier"
Using well-trained DNN, SVM, Adaboost, Decision_tree, Gradient_boosting, KNN, LightGBM, Random Forest models to predict 0-1.0 Ga independent test sets
1. Please start with main file: "code/main.py", training models and predicting mantle source types of unknown basalts.
2. All well-trained models files in the filefolder "model/"  for 8 mainstream models including DNN, SVM, Adaboost, Decision_tree, Gradient_boosting, KNN, LightGBM, Random Forest models, are called in main.py by changing method="DNN", or "SVM", or other models.
3. Other figures in the paper shall be easy to reproduce by running "main.py".
4. File path and some parameters are defined in "globalVar.py".
5. Functions for plot figures and normalization are defined in "utils.py".

# Running the project 
## In terminal for Ubuntu or command-line interface for windows
Firstly, clone this project.
```
git clone https://github.com/jiangjilian/mantle_source_classifier.git
```
Then, to install dependent libraries for our project,run
```
cd mantle_source_classifier
pip install -r requirements.txt
```
And then, run the main.py to obtain prediction results and our figures.
```
cd code
python main.py
```
## In Pycharm
Install all dependent libraries and directly run main.py
  
