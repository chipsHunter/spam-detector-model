
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from   sklearn import metrics
from tfidf import get_tfidf, clean_text
from   sklearn.ensemble import RandomForestClassifier
import pandas as pd


X_train, X_test, y_train, y_test, vectorizer, le = get_tfidf()

import os
os.environ['TERM'] = 'xterm'
os.system('clear')

def get_forest():
    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, y_train)

    dataframe = pd.DataFrame(columns=['Feature', 'FeatureImportance'])
    dataframe['Feature'] = X_train.columns
    dataframe['FeatureImportance'] = random_forest.feature_importances_
    print('Importance of the Features in the Model\n\n', dataframe.sort_values(by='FeatureImportance', ascending=False))
    return random_forest

if __name__ == '__main__':
    forest = get_forest()
    yTrainPred = forest.predict(X_train)

    confusionMatrix = confusion_matrix(y_train, yTrainPred, labels=forest.classes_)
    train = pd.DataFrame(confusionMatrix, columns=forest.classes_)
    train.index = forest.classes_
    print('Confusion Matrix\n\n', train)
    print("\n\nTraining Accuracy = %5.5f%s" % (metrics.accuracy_score(y_train, yTrainPred) * 100, "%"))

    yTestPred = forest.predict(X_test)

    confusionTestMatrix = confusion_matrix(y_test, yTestPred, labels=forest.classes_)
    test = pd.DataFrame(confusionMatrix, columns=forest.classes_)
    test.index = forest.classes_
    print('Confusion Matrix\n\n', test)
    print("\n\nTest Accuracy = %5.5f%s" % (metrics.accuracy_score(y_test, yTestPred) * 100, "%"))

    sk_roc_auc = roc_auc_score(y_test, yTestPred)
    sk_fpr, sk_tpr, sk_thresholds = roc_curve(y_test, yTestPred)

    print(f'\nAUC (scikit-learn): {sk_roc_auc}', '', sep='\n')
    print('Optimal False Positive Rates (scikit-learn):', sk_fpr, '', sep='\n')
    print('Optimal True Positive Rates (scikit-learn):', sk_tpr, '', sep='\n')
    print('Optimal thresholds (scikit-learn):', sk_thresholds, sep='\n')


