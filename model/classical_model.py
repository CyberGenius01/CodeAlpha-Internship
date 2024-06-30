## IMPORTING LIBRARIES
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc

## IMPORTING MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

## INITIALZATION OF PARAMETERS
DATASET_PATH = os.path.join(os.path.curdir, 'Titanic-Dataset.csv')
REMOVED_COLS = ['PassengerId', 'Name', 'Ticket', 'Cabin']

## IMPORTING DATASET
dataset = pd.read_csv(DATASET_PATH)
ds = dataset.drop(REMOVED_COLS, axis=1)
X = ds.iloc[:,1:].values
y = ds.iloc[:,0].values

## HANDLING MISSING VALUES IN NUMERIC COLUMNS
col_numbers = [0,2,3,4,5]
imputer = SimpleImputer(missing_values=np.nan, strategy='mean', fill_value='float')
X[:,col_numbers] = imputer.fit_transform(X[:,col_numbers])

## ENCODING CATEGORICAL DATA USING ONE HOT ENCODER
col_cat = [1,6]
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), col_cat)],
    remainder='passthrough'
)
X = np.array(ct.fit_transform(X))

## NORMALIZATION (GENERALLY NOT USED FOR REGRESSION)
sc = StandardScaler()
X = sc.fit_transform(X)

## SPLITTING THE DATASET INTO TRAIN AND TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=123)

## MODEL TESTING

### Logistic Regression
lrg = LogisticRegression(penalty='l2', random_state=42, solver='newton-cholesky')
lrg.fit(X_train, y_train)
lrg_pred = lrg.predict(X_test)

### K-Nearest Neighbours
knn = KNeighborsClassifier(n_neighbors=12, weights='distance', algorithm='kd_tree', metric='minkowski')
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

### Support Vector Classifier
svc = SVC(kernel='rbf', degree=5, gamma='scale', random_state=123)
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)

### Naive Bayes Classification
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_pred = gnb.predict(X_test)

### Decision Tree 
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=100, max_features=150, random_state=123)
dtc.fit(X_train, y_train)
dtc_pred = dtc.predict(X_test)

### Random Forest
rfc = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=50, max_features=150, random_state=123)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

preds = np.concatenate((np.array(lrg_pred).reshape(-1,1),
                        np.array(knn_pred).reshape(-1,1),
                        np.array(svc_pred).reshape(-1,1),
                        np.array(gnb_pred).reshape(-1,1),
                        np.array(dtc_pred).reshape(-1,1),
                        np.array(rfc_pred).reshape(-1,1)),
                      axis=1)

def plot_roc(pred):
    fpr, tpr, roc_auc, classes = dict(), dict(), dict(), 6
    
    for i in range(classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    plt.style.use('dark_background')    
    fig, ax = plt.subplots(2,1,figsize=(7,5), gridspec_kw={'height_ratios': [0.8, 100]})
    fig.tight_layout(pad=3.0)
    fig.suptitle('Comaprison of ROC Curves of Models')
    colors = ['cyan', 'magenta', 'dodgerblue', 'seagreen', 'chocolate', 'orange']
    models = ['LRG', 'KNN', 'SVC', 'GNB', 'DTC', 'RFC']
    legends = []
    for i in range(classes):
        ax[1].plot(fpr[i], tpr[i], color=colors[i], lw=1.2)
        ax[1].grid(color='gray', linestyle='--', linewidth=0.7)
        legends.append(f'ROC curve (area = {roc_auc[i]:.2f}) for {models[i]}')
    x_range = np.linspace(0,1,50)
    ax[1].plot(x_range, x_range, color='white', linestyle='--', linewidth=1.2)
    fig.legend(legends, bbox_to_anchor=(0.935,0.93), ncols=2)
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[0].axis('off')
    with open('C:\\Users\\ritesh\\Desktop\\CodeSoft\\model\\config.json', 'r') as f:
        data = json.load(f)
    files = data[0]['files']
    plt.savefig(files['roc'],transparent=True)
    
    best_model = [lrg, knn, svc, gnb, dtc, rfc]
    index = list(roc_auc).index(max(list(roc_auc)))
    return best_model[index], models[index]
    
def plot_graphs(model, model_name):
    labels = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    print(f"Shape of X_train before inverse transform: {X_train.shape}")
    
    X_set, y_set = sc.inverse_transform(X_train), y_train
    print(f"Shape of X_train after inverse transform: {X_set.shape}")

    with open('C:\\Users\\ritesh\\Desktop\\CodeSoft\\model\\config.json', 'r') as f:
        data = json.load(f)
    files = data[0]['files']
    
    for x in range(len(labels)):
        for n in range(len(labels) - x):
            X1, X2 = np.meshgrid(
                np.arange(start=X_set[:, 6+x].min() - 10, stop=X_set[:, 6+x].max() + 10, step=0.25),
                np.arange(start=X_set[:, 6+n].min() - 1000, stop=X_set[:, 6+n].max() + 1000, step=0.25)
            )

            plt.style.use('dark_background')
            
            # Ensure the input to sc.transform has the correct number of features
            input_data = np.array([X1.ravel(), X2.ravel()]).T
            if input_data.shape[1] != X_train.shape[1]:
                # Adjust the shape of the input data if needed
                adjusted_input_data = np.zeros((input_data.shape[0], X_train.shape[1]))
                adjusted_input_data[:, 6+x] = X1.ravel()
                adjusted_input_data[:, 6+n] = X2.ravel()
                input_data = adjusted_input_data

            transformed_data = sc.transform(input_data)
            
            plt.contourf(
                X1, X2, model.predict(transformed_data).reshape(X1.shape),
                alpha=0.75, cmap=ListedColormap(('red', 'green'))
            )
            plt.xlim(X1.min(), X1.max())
            plt.ylim(X2.min(), X2.max())
            for i, j in enumerate(np.unique(y_set)):
                plt.scatter(
                    X_set[y_set == j, 6+x], X_set[y_set == j, 6+n],
                    c=ListedColormap(('red', 'green'))(i), label=j
                )
            plt.title(f'{model_name} (Training set)')
            plt.xlabel(f'{labels[x]}')
            plt.ylabel(f'{labels[n]}')
            plt.legend()

            plot_file_path = f"C:\\Users\\ritesh\\Desktop\\CodeSoft\\model\\static\\media\\best_model{x+n}.png"
            files[f'best_model{x+n}'] = plot_file_path
            plt.savefig(plot_file_path)
            plt.clf()  # Clear the plot for the next iteration
            
    with open('C:\\Users\\ritesh\\Desktop\\CodeSoft\\model\\config.json', 'w') as f:
        json.dump(data, f, indent=4)
        
    return x*n