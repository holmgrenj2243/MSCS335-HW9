import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

#https://www.kaggle.com/datasets/yekenot/tree-survival-prediction
df = pd.read_csv("Tree_Data.csv")
df = df.fillna(0)
df.Alive = df.Alive.replace('X', 1)
df[df.select_dtypes('int64').columns] = df.select_dtypes('int64').astype('float64')

#Predicting if tree is alive 1=alive 0=dead
#Light_ISF=Light level reaching each subplot
#AMF=percent of arbuscular mycorrhizal fungi in root of havested seeds
#Phenolics=Gallic acid equivalents(nmol) per mg of dry extract
#NSC=Percent of dry mass nonstructured carbohydrates
df = df[['Light_ISF', 'AMF','Phenolics','Lignin','NSC',"Alive"]]

X = df.iloc[:, :-1].to_numpy()
y = df.iloc[:, -1].to_numpy()
X = (X - np.average(X, axis=0))/np.std(X, axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = SVC()
parameters = {"C": np.linspace(1, 50, num=10)}
grid_search = GridSearchCV(clf, param_grid = parameters, scoring = "accuracy")
grid_clf = grid_search.fit(X_train, y_train)
score_df = pd.DataFrame(grid_search.cv_results_)
print(score_df[['param_C', 'mean_test_score', 'rank_test_score']])

C = grid_search.best_params_['C']
clf = SVC(C=C)
clf.fit(X_train, y_train)

print(f"Score: {clf.score(X_test, y_test):.3f}")
cm = confusion_matrix(y_test, clf.predict(X_test)) #normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()