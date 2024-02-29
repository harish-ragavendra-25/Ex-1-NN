<H3>ENTER YOUR NAME HARISH RAGAVENDRA S</H3>
<H3>ENTER YOUR REGISTER NO. 212222230045</H3>
<H3>EX. NO.1</H3>
<H3>DATE 29-02-2024</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("/content/Churn_Modelling.csv")
print(df)
df.head()
```
```
X=df.iloc[:,:-1].values
print(X)

y=df.iloc[:,-1].values
print(y)
```
```
print(df.isnull().sum())

df.duplicated()

print(df['HasCrCard'].describe())
```
```
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()
```
```
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
```
```
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))

print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```


## OUTPUT:
![Screenshot 2024-02-29 190346](https://github.com/harish-ragavendra-25/Ex-1-NN/assets/114852180/80aeba64-938f-4126-8e59-0372ca22733e)
![Screenshot 2024-02-29 190351](https://github.com/harish-ragavendra-25/Ex-1-NN/assets/114852180/e52b8ada-b2f9-4d34-831e-e765b60ab470)
![Screenshot 2024-02-29 190357](https://github.com/harish-ragavendra-25/Ex-1-NN/assets/114852180/54378abb-a899-4f16-ab3b-082706e6dead)
![Screenshot 2024-02-29 190413](https://github.com/harish-ragavendra-25/Ex-1-NN/assets/114852180/8e2c80b5-a849-4dcf-bd33-c8af8a348302)
![Screenshot 2024-02-29 190429](https://github.com/harish-ragavendra-25/Ex-1-NN/assets/114852180/52bf22bc-ddf7-4f42-adee-1c7834dacc48)
![Screenshot 2024-02-29 190436](https://github.com/harish-ragavendra-25/Ex-1-NN/assets/114852180/15277847-3a88-4c93-8542-8964d2668904)
![Screenshot 2024-02-29 190448](https://github.com/harish-ragavendra-25/Ex-1-NN/assets/114852180/b8dba549-e6af-46bd-b731-6ea11c9fb912)
![Screenshot 2024-02-29 190500](https://github.com/harish-ragavendra-25/Ex-1-NN/assets/114852180/8b3c6318-1a99-4d9c-ac29-784750bc3c96)
![Screenshot 2024-02-29 190507](https://github.com/harish-ragavendra-25/Ex-1-NN/assets/114852180/d9d04b8d-f22f-44df-806a-3db0e327e809)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


