import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("dataset.xlsx")

print(df.isnull().sum())

df =df.dropna(how='any',axis='rows')
print(df.shape)

df["Total Price"] = df["Quantity"] * df["UnitPrice"]
print(df.head())

#number of order by country
ord_cntry = df.groupby('Country')['InvoiceNo'].count().sort_values(ascending=False)


#plot
plt.figure(figsize=(12,6))
ord_cntry.plot.bar(logy=True)
plt.xlabel('Country') 
plt.ylabel('No Of Orders')
plt.title('Number of orders per country')
plt.show()

#Amout spend per country
amt_cntry = df.groupby('Country')['Total Price'].sum().sort_values(ascending=False)


#plot
plt.figure(figsize=(12,6))
amt_cntry.plot.bar(logy=True)
plt.xlabel('Country') 
plt.ylabel('Total amount spend')
plt.title('Total amount spend per country')
plt.show()