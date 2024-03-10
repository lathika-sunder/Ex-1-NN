<H3> NAME : Lathika Sunder</H3>
<H3>REGISTER NO. : 212221230054</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle.

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
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df=pd.read_csv('Churn_Modelling.csv')
print(df)
df.head()
X=df.iloc[:,:-1].values
print(X)
y=df.iloc[:,-1].values
print(y)

print(df.isnull().sum())
df.duplicated()

print(df['CustomerId'].describe())
print(df['Surname'].describe())
print(df['CreditScore'].describe())
print(df['Geography'].describe())
print(df['Gender'].describe())
print(df['Age'].describe())
print(df['Tenure'].describe())
print(df['Balance'].describe())
print(df['NumOfProducts'].describe())
print(df['HasCrCard'].describe())
print(df['IsActiveMember'].describe())
print(df['EstimatedSalary'].describe())
print(df['Exited'].describe())

data = df.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```

## OUTPUT:
### Dataframe:
![image](https://github.com/Janani-2003/Ex-1-NN/assets/94288340/4dd4cecf-d41c-4271-967b-95475e2dce8f)
![image](https://github.com/Janani-2003/Ex-1-NN/assets/94288340/1c90dac5-7ad8-41eb-a3ad-d248c8bd538a)
### Values of X:
![image](https://github.com/Janani-2003/Ex-1-NN/assets/94288340/d56f1da9-6831-4691-a1b4-18795d4964ed)
### Values of Y:
![image](https://github.com/Janani-2003/Ex-1-NN/assets/94288340/f5c42cd0-42ca-43a0-acc3-cf372eb443d2)
### Null values:
![image](https://github.com/Janani-2003/Ex-1-NN/assets/94288340/a46d3706-7d4a-4bbd-9d49-3706b5085678)

![image](https://github.com/Janani-2003/Ex-1-NN/assets/94288340/a2657627-d999-4cec-b262-8c768a0b8a98)
_______________________________________________________________________________________________________

![image](https://github.com/Janani-2003/Ex-1-NN/assets/94288340/9c28da72-8c7d-4254-92bc-909a19648ec2)
_______________________________________________________________________________________________________

![image](https://github.com/Janani-2003/Ex-1-NN/assets/94288340/540754f3-85bc-44d0-bcef-dc4dfb0bf396)
_______________________________________________________________________________________________________

![image](https://github.com/Janani-2003/Ex-1-NN/assets/94288340/269e98cf-fc52-4a91-ae90-7a830ceb349b)
_______________________________________________________________________________________________________

![image](https://github.com/Janani-2003/Ex-1-NN/assets/94288340/e7fe5227-1664-4c95-9574-249b5afa151d)
_______________________________________________________________________________________________________

![image](https://github.com/Janani-2003/Ex-1-NN/assets/94288340/4768dc4d-72da-4c6c-a532-5f27d703f8eb)
_______________________________________________________________________________________________________

![image](https://github.com/Janani-2003/Ex-1-NN/assets/94288340/bcd20978-17af-4d23-a91b-0ad67379f4ad)
_______________________________________________________________________________________________________

![image](https://github.com/Janani-2003/Ex-1-NN/assets/94288340/47433259-a884-4d1a-94d6-65d22745ca3e)
_______________________________________________________________________________________________________

![image](https://github.com/Janani-2003/Ex-1-NN/assets/94288340/c280f53b-0b00-4cb8-a5a0-2c46f25e127a)
_______________________________________________________________________________________________________

![image](https://github.com/Janani-2003/Ex-1-NN/assets/94288340/92dec003-191c-4e7e-87c2-6411f9a2d179)
_______________________________________________________________________________________________________

![image](https://github.com/Janani-2003/Ex-1-NN/assets/94288340/c796bf07-bbdb-4a4a-b10f-8e161a21f01d)
_______________________________________________________________________________________________________

![image](https://github.com/Janani-2003/Ex-1-NN/assets/94288340/ad5cc7b4-44d0-4759-a584-07eefb68e821)
_______________________________________________________________________________________________________

![image](https://github.com/Janani-2003/Ex-1-NN/assets/94288340/7e59a11b-3f22-4d6e-88de-ed8a8223e64d)
_______________________________________________________________________________________________________

![image](https://github.com/Janani-2003/Ex-1-NN/assets/94288340/a25013c0-1cac-4d1d-8e05-e432dda66eae)
_______________________________________________________________________________________________________
### MinMax Scaling:
![image](https://github.com/Janani-2003/Ex-1-NN/assets/94288340/43e9d81f-e928-4c37-973c-81af84a7c431)
_______________________________________________________________________________________________________
![image](https://github.com/Janani-2003/Ex-1-NN/assets/94288340/08d51576-7a34-4617-aa02-698015202672)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


