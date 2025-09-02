# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
        
import pandas as pd
df=pd.read_csv("C:\\Users\\admin\\Downloads\\titanic_dataset (1).csv")
df

<img width="1237" height="441" alt="image" src="https://github.com/user-attachments/assets/c00449b8-ed55-4030-8b5c-79b8cbf9836b" />

df.shape

<img width="115" height="43" alt="image" src="https://github.com/user-attachments/assets/f46a04e5-27e2-488f-9ef1-fc4a933a2d60" />

df.set_index("PassengerId",inplace=True)
df

<img width="1220" height="471" alt="image" src="https://github.com/user-attachments/assets/96ad136b-7e1c-414e-81bb-80447a189d37" />

df.Pclass.unique()

<img width="297" height="37" alt="image" src="https://github.com/user-attachments/assets/9a6b7652-b946-4d8a-b660-d4f1a28ac2ef" />


df.nunique()

<img width="238" height="271" alt="image" src="https://github.com/user-
attachments/assets/5ec73ea9-e9bc-43b0-8308-18201592dd13" />


df['Survived'].value_counts()

<img width="346" height="78" alt="image" src="https://github.com/user-attachments/assets/65a65834-fc12-49d6-a564-c1d8660f6829" />


df.Survived.unique()

<img width="276" height="46" alt="image" src="https://github.com/user-attachments/assets/adbf9f05-a5d9-44ff-aed1-d56f9028194b" />

df.rename(columns={"Sex":"Gender"},inplace=True)
df

<img width="1226" height="441" alt="image" src="https://github.com/user-attachments/assets/a4629503-48f1-412b-8d50-e87369ced493" />

import seaborn as sns
sns.countplot(data=df)

<img width="907" height="572" alt="image" src="https://github.com/user-attachments/assets/cd93de42-09ba-419e-88d6-977eb311d616" />


sns.countplot(x="Survived",hue="Gender",data=df)

<img width="791" height="577" alt="image" src="https://github.com/user-attachments/assets/9a3bd933-616a-4444-9395-daafcbf7354c" />


sns.catplot(x="Survived",hue="Gender",data=df,kind="violin")

<img width="925" height="625" alt="image" src="https://github.com/user-attachments/assets/869afa46-d2af-4543-a0a9-7257cda46ad1" />


sns.boxplot(data=df)

<img width="772" height="557" alt="image" src="https://github.com/user-attachments/assets/9f8df431-540e-440d-bba7-b5ee88c20521" />


df.boxplot(column="Survived",by="Gender")

<img width="803" height="617" alt="image" src="https://github.com/user-attachments/assets/48673b00-9f10-42cf-bf02-1dffa1695698" />

sns.scatterplot(data=df)

<img width="765" height="583" alt="image" src="https://github.com/user-attachments/assets/54697952-6567-4b9e-8975-e6fc37d66465" />


sns.scatterplot(x=df['Age'],y=df['Fare'])

<img width="867" height="582" alt="image" src="https://github.com/user-attachments/assets/1974268f-1517-4cc9-ab6c-89febe66e397" />


sns.jointplot(x="Age",y='Fare',data=df)

<img width="812" height="776" alt="image" src="https://github.com/user-attachments/assets/77f7decb-f3aa-4337-ae5c-0982c3c57204" />


sns.jointplot(x="Age",y='Fare',data=df,kind='kde')

<img width="796" height="782" alt="image" src="https://github.com/user-attachments/assets/dd37dfe7-7d10-4551-9c19-3dccbe4a57e5" />


sns.jointplot(x="Age",y='Fare',data=df,kind='hex')

<img width="763" height="777" alt="image" src="https://github.com/user-attachments/asset<img width="1238" height="620" alt="image" src="https://github.com/user-attachments/assets/f4085b41-f930-485f-84d0-2f2c9820973b" />
s/2403117f-66b3-44d3-a4f8-14eaac7afdcb" />


sns.pairplot(data=df)

<img width="1237" height="647" alt="image" src="https://github.com/user-attachments/assets/d655c855-3ff4-46e2-b2ac-34afdd9adabe" />
<img width="655" height="555" alt="image" src="https://github.com/user-attachments/assets/22105b38-ca0b-454e-b17e-8e7a4f5f561b" />


corr1=df.select_dtypes(include=['number']).corr()
sns.heatmap(corr1,annot=True)

<img width="1257" height="657" alt="image" src="https://github.com/user-attachments/assets/ce0c4a93-5ed8-43b9-aba5-659c843b2584" />


sns.catplot(x='Survived',col='Gender',data=df,kind='count')

<img width="843" height="612" alt="image" src="https://github.com/user-attachments/assets/a63dd765-b357-4abf-bb0f-037907b5d264" />


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


titanic = sns.load_dataset('titanic')

cols = ['age', 'fare']
df = titanic[cols].dropna()


for col in cols:
    print(f"{col.capitalize()} → Skewness: {skew(df[col]):.3f}, Kurtosis: {kurtosis(df[col]):.3f}")

corr = titanic[['survived','pclass','age','fare']].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (Titanic)")
plt.show()


metrics = pd.DataFrame({
    "Feature": cols,
    "Skewness": [skew(df[c]) for c in cols],
    "Kurtosis": [kurtosis(df[c]) for c in cols]
})


metrics.set_index("Feature").plot(kind="bar", figsize=(8,5))
plt.title("Skewness & Kurtosis of Titanic Data")
plt.ylabel("Value")
plt.xticks(rotation=0)
plt.show()

<img width="906" height="575" alt="image" src="https://github.com/user-attachments/assets/40706cb9-5a58-4c80-8e1b-3a3504aa09ac" />
![Uploading image.png…]()

# RESULT
the analysis is compleated successfully
