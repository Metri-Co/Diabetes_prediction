#virutal assistant: metri ojeda J.C
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


def remove_ids(array, var_position):
    ids = array[:, var_position]
    array = np.delete(array, var_position, axis=1)
    vector = np.ravel(array, "F")
    return vector, ids


def accuracy(groundtruth, predictions):
    tn, fp, fn, tp = confusion_matrix(groundtruth, predictions, labels=[0, 1]).ravel()
    obs = len(groundtruth)
    result = (tp + tn) / obs
    return result


def precision(groundtruth, predictions):
    # true positives / true positives + false positives
    tn, fp, fn, tp = confusion_matrix(groundtruth, predictions).ravel()
    result = tp / (tp + fp)
    return result


def recall(groundtruth, predictions):
    tn, fp, fn, tp = confusion_matrix(groundtruth, predictions).ravel()
    result = tp / (tp + fn)
    return result


def F1(groundtruth, predictions):
    numerator = precision(groundtruth, predictions) * recall(groundtruth, predictions)
    denominator = precision(groundtruth, predictions) + recall(groundtruth, predictions)
    result = 2 * (numerator / denominator)
    return result


def create_metrics(groundtruth, predictions):
    dic = {"accuracy": accuracy(groundtruth, predictions),
           "precision": precision(groundtruth, predictions),
           "recall": recall(groundtruth, predictions),
           "Fvalue": F1(groundtruth, predictions)}
    return dic
def normalize(df):
    import pandas as pd
    norm_df = df.copy()
    _, y = norm_df.shape

    for i in range(y):
        min_i = min(norm_df.iloc[:,i])
        max_i = max(norm_df.iloc[:,i])

        norm_df = norm_df.apply(lambda x: (x - min_i)/(max_i - min_i), axis = 1)

    return norm_df



def roc_curve(groundtruth, probabilities, predictions, estimator_name=str):
    """
    Function for plotting the ROC curve and calculating the AUC

    Parameters
    ----------
    groundtruth : List or 1D array
        The real values of the dataset.
    predictions : List or 1D array
        The predicted values by the classifier.
    estimator_name : string
        Name of the classifier, it will be printed in the Figure

    Returns
    -------
    Figure.
    AUC.
    """
    from sklearn.metrics import roc_auc_score
    sensitivities = []
    especificities = []

    sensitivities.append(1)
    especificities.append(1)

    thresholds = [i * 0.05 for i in range(1, 10, 1)]
    for t in thresholds:
        prob = probabilities[:, 1]
        prob = np.where(prob >= t, 1, 0)
        recall_data = recall(groundtruth, prob)
        precision_data = precision(groundtruth, prob)
        sensitivities.append(recall_data)
        espc = 1 - precision_data
        especificities.append(espc)
    sensitivities.append(0)
    especificities.append(0)

    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(especificities, sensitivities, marker='o', linestyle='--', color='r')
    plt.plot([i * 0.01 for i in range(100)], [i * 0.01 for i in range(100)])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title(f'{estimator_name} ROC curve')
    plt.savefig(f'{estimator_name} ROC curve.jpg', dpi=300)

    AUC = roc_auc_score(groundtruth, predictions)
    return AUC


def add_gnoise(df, target_loc, target_class):
    ### filter for the desired class##
    dfc = df.copy()
    cols = dfc.columns.tolist()
    dfc = dfc[dfc[target_loc] == target_class]
    dfc = dfc.drop([target_loc], axis = 1)


    shape = dfc.shape
    dfc = np.array(dfc).reshape(shape)

    noise = np.random.normal(0, 0.1, shape)
    new_df = dfc + noise

    if target_class == 1:
        label_col = np.ones((shape[0], 1))

    elif target_class == 0:
        label_col = np.zeros((shape[0], 1))

    new_df = np.concatenate((new_df, label_col), axis=1)
    # creating a list of index names
    index_values = [i for i in range(shape[0])]

    new_df = pd.DataFrame(data=new_df,
                      index=index_values,
                      columns=cols)
    df = pd.concat([df, new_df], join = 'outer')
    return df

def make_mi_scores(X, y, target = 'reg'):
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.feature_selection import mutual_info_regression
    if target == 'reg':
        mi_scores = mutual_info_regression(X, y, discrete_features='auto')
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns.tolist())
        mi_scores = mi_scores.sort_values(ascending=False)
    if target == 'cat':
        mi_scores = mutual_info_classif(X, y, discrete_features='auto')
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns.tolist())
        mi_scores = mi_scores.sort_values(ascending=False)

    return mi_scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    plt.tight_layout()
    plt.savefig('Mutual information.jpg', dpi = 300)

def plot_features(df, features = list, label_name = str):

    rows = int(np.ceil(len(features)/2))
    n_features = len(features)
    cols = 2
    fig, axs = plt.subplots(rows,cols, figsize =(12,12), dpi = 300)

    colors = {0: 'blue',
              1: 'red'}
    ### col 1 ###
    count = 0
    for i in range(0, rows):
        axs[i,0].scatter(df[features[count]], df[features[count + 1]],
                             c = df[label_name].map(colors))
        axs[i,0].set_ylabel(features[count + 1])
        axs[i,0].set_xlabel(features[count])
        count += 1
    for i in range(0, rows -1):
        axs[i,1].scatter(df[features[count]], df[features[count + 1]],
                             c = df[label_name].map(colors))
        axs[i,1].set_ylabel(features[count + 1])
        axs[i,1].set_xlabel(features[count])

        count += 1
    plt.savefig('features visualization.jpg', dpi = 300)


##########################################
##########################################
# Raw data with just feature engineering #
##########################################
##########################################

data = pd.read_csv(r'Diabetes/diabetes_anthro.csv')
comma_cols = ['chol_hdl_ratio','bmi','waist_hip_ratio']
continuous_feat = data.columns.tolist()[:-1]
categorical_feat = data.columns.tolist()[-1]

plt.hist(np.array(data.loc[:,'diabetes']).flatten())
plt.show()

for i in range(data.shape[0]):
    if data.iloc[i, -1] == 'No diabetes':
        data.iloc[i, -1] = 0
    elif data.iloc[i, -1] == 'Diabetes':
        data.iloc[i, -1] = 1

for i in range(data.shape[0]):
    if data.iloc[i, 6] == 'female':
        data.iloc[i, 6] = 0
    elif data.iloc[i, 6] == 'male':
        data.iloc[i, 6] = 1

for col in comma_cols:
    data[col] = data[col].str.replace(',', '.')

for i in range(len(continuous_feat)):
    col = continuous_feat[i]
    data[col] = pd.to_numeric(data[col], errors = 'raise')

data['diabetes'] = data['diabetes'].astype('int64')

new_col =  data['systolic_bp'] * data['diastolic_bp']
data.insert(loc= 15, column = 'syst_diast_interaction', value = new_col)

new_col =  data['hip'] * data['waist']
data.insert(loc= 16, column = 'hip_waist_interaction', value = new_col)

new_col =  data['age'] * data['bmi']
data.insert(loc= 17, column = 'age_bmi_interaction', value = new_col)

new_col =  data['systolic_bp'] * data['bmi']
data.insert(loc= 18, column = 'syst_bmi_interaction', value = new_col)

data = data.drop(['patient_number'], axis = 1)

X = data.loc[:, features]
y = data.iloc[:,14]

features_info = make_mi_scores(X, y,'cat')
plt.figure(dpi=300, figsize=(8, 8))
plot_mi_scores(features_info)

features = [features_info.index.tolist()[i] for i in range(len(features_info)) if features_info[i] >= 0.01]
X = np.array(data.loc[:,features])
y = np.array(y)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
####### Logistic regression model #######
logit = LogisticRegression(solver = 'lbfgs', penalty = 'l2')
log_scores = cross_val_score(logit, X, y,cv = 10)

logit = logit.fit(X_train, y_train)
logit_proba = logit.predict_proba(X_test)
logit_y_pred = logit.predict(X_test)

logit_metrics = create_metrics(y_test, logit_y_pred)
logit_auc = roc_curve(y_test, logit_proba, logit_y_pred, estimator_name= 'Logistic Regression classifier')


####### KNN classifier #######

param_grid = {'n_neighbors': [3,5,7,9,11,13,15],
              'weights': ['uniform', 'distance'],
              'algorithm': ['kd_tree', 'brute'],
              'p': [1,2]}

grid = GridSearchCV(KNeighborsClassifier(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)
print(grid.best_estimator_)

knn_clf = KNeighborsClassifier(n_neighbors= 3, weights = 'uniform'
                               ,algorithm = 'kd_tree', p = 1)
knn_scores = cross_val_score(knn_clf, X, y, cv = 10)

knn_clf =knn_clf.fit(X_train, y_train)

knn_y_pred = knn_clf.predict(X_test)
knn_proba = knn_clf.predict_proba(X_test)

knn_metrics = create_metrics(y_test, knn_y_pred)
knn_auc = roc_curve(y_test, knn_proba, knn_y_pred, estimator_name= 'KNN classifier')

####### Decision tree #######
param_grid = {'min_samples_leaf': [1,2, 3, 4, 5, 6],
              'max_depth': [1,2,3,4,5,6, None],
              'criterion': ['entropy', 'gini']}

grid = GridSearchCV(DecisionTreeClassifier(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)
print(grid.best_estimator_)

clf = DecisionTreeClassifier(min_samples_leaf=5, max_depth= 3, criterion = 'entropy')
clf_scores = cross_val_score(clf, X, y, cv = 10)
clf = clf.fit(X_train, y_train)

clf_y_pred = clf.predict(X_test)
clf_proba = clf.predict_proba(X_test)

clf_metrics = create_metrics(y_test, clf_y_pred)

####### Evaluation with barcharts #######

x_axis = np.arange(0,3)

accuracies = [logit_metrics['accuracy'], knn_metrics['accuracy'], clf_metrics['accuracy']]
precisions = [logit_metrics['precision'], knn_metrics['precision'], clf_metrics['precision']]
recalls = [logit_metrics['recall'], knn_metrics['recall'], clf_metrics['recall']]


plt.figure(figsize = (8,5), dpi = 300)
plt.bar(x=x_axis - 0.25, height = accuracies, width = 0.25, label ='accuracy', color = 'red')
plt.bar(x=x_axis, height = precisions, width = 0.25, label ='precision', color = 'blue')
plt.bar(x=x_axis + 0.25, height = recalls, width = 0.25, label ='recall', color = 'orange')

plt.ylabel('Performance (%)')
plt.legend(loc = 4)
plt.xticks(x_axis, ['Logit Class', 'KNN', 'DT'])
plt.title("Models' performance without white noise")
plt.ylim([0,1.1])
#plt.show()
plt.savefig('Diabetes/Non engineered models.jpg', dpi =300)

