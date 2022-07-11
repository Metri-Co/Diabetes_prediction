# Diabetes_prediction
Hi! Here I developed Machine Learning models for diabetes II prediction. The dataset was taken from Kaggle:
https://www.kaggle.com/datasets/houcembenmansour/predict-diabetes-based-on-diagnostic-measures

You can take the programm with no data augmentation `` or the augmente `` to reproduce or improve this work.

## Motivation
The main motivation of this work was to practice feature engineering, data augmentation, feature selection, and model selection. Moreover, I am a Nutritionist, so I was very excited for creating a possible auxiliar tool for my colleagues. One of the main reasons why I selected this dataset is because it use normal anthropometric and biochemical tests that most of the nutritionist can have in handly.

## Data exploration
First, most of the data type should be converted from 'Object' to 'Nuerical' or 'Float', also, some of the float type features are separated by commas instead of dots, so, you can do these conversions using the following lines of code
```
data = pd.read_csv(r'Diabetes/diabetes_anthro.csv')
comma_cols = ['chol_hdl_ratio','bmi','waist_hip_ratio']
continuous_feat = data.columns.tolist()[:-1]
categorical_feat = data.columns.tolist()[-1]

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
```
I manage the labels as `int64` but you can manage them as `category`

As it is widely reported, glucose is one of the key indicators for diabetes diagnosis, so, I wanted to create new variable or features that could complement this feature. You can use the same features interactions as I did or you can create new ones. I selected hip-waist, bmi, age, and diastolyc/systolic because the are the most related features to diabetes risk.

```
new_col =  data['systolic_bp'] * data['diastolic_bp']
data.insert(loc= 15, column = 'syst_diast_interaction', value = new_col)

new_col =  data['hip'] * data['waist']
data.insert(loc= 16, column = 'hip_waist_interaction', value = new_col)

new_col =  data['age'] * data['bmi']
data.insert(loc= 17, column = 'age_bmi_interaction', value = new_col)

new_col =  data['systolic_bp'] * data['bmi']
data.insert(loc= 18, column = 'syst_bmi_interaction', value = new_col)
```
Now, there is a non useful column, the patient id, so you can get rid of that using `data = data.drop(['patient_number'], axis = 1)`

As you can see, this is an imbalanced dataset, as it has only 60 positive observations vs 330 negative observations

![Original dataframe](https://user-images.githubusercontent.com/87657676/178353431-47526eeb-1cce-442a-a426-46a66b1e9a2f.jpg)

After this, I used the `add_gnoise`function for the data augmentation and now the classes are more balanced

![Augmented dataframe](https://user-images.githubusercontent.com/87657676/178353520-ece107a0-52a5-4e72-b950-22a40ef0a205.jpg)

## Feature selection

I used Mutual Information because it is useful for non-linear responses. So, you can use the following code (all of them are functions)
```
X = data.iloc[:, :-1]
y = data.iloc[:,-1]

features_info = make_mi_scores(X, y,'cat')
plt.figure(dpi=300, figsize=(8, 8))
plot_mi_scores(features_info)

features = [features_info.index.tolist()[i] for i in range(len(features_info)) if features_info[i] >= 0.30]
X = np.array(data.loc[:,features])
y = np.array(y)

```

![Mutual information](https://user-images.githubusercontent.com/87657676/178353910-9f3cc22a-81bf-4511-9205-687be27c2946.jpg)

Feel free to change the threshold of the MI index in `features` list

## Model evaluation
I used a simple Logistic Regression model, a KNN, and a Decision Tree. The imbalance of the original dataset showed accurate models (~ 85 %of accuracy) but very low recall, which is undesired in medical applications.

![Non engineered models](https://user-images.githubusercontent.com/87657676/178354136-8b01ba33-903d-4824-a049-44f37a0c4d30.jpg)

However, the data augmentation leads to an incredible increase of recall, and accuracy and precision were also improved.

![Diabetes engineered models](https://user-images.githubusercontent.com/87657676/178354247-c5dea942-b331-49d2-8e77-ad27ec5b48c1.jpg)

Hope you enjoyed these models! You can find all the code in the scripts, this was just a summary.
