#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#dataset
df = pd.read_excel('dataset.xlsx')


df = df[df['CustomerID'].notna()]

#creating RFM Table for Segmentation purpose

#converting date_time to show only date
from datetime import datetime
df['InvoiceDate'] = df['InvoiceDate'].dt.date

#total sum
df['TotalSum'] = df['Quantity'] * df['UnitPrice']

#data variable to store recency count
import datetime
snapshot_date = max(df.InvoiceDate) + datetime.timedelta(days = 1)

#aggregating
customers = df.groupby(['CustomerID']).agg({
    'InvoiceDate' : lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo' : 'count',
    'TotalSum' : 'sum'
})

#renaming columns
customers.rename(columns = {
    'InvoiceDate': 'Recency',
    'InvoiceNo' : 'Frequency',
    'TotalSum' : 'MonetaryValue'
},
    inplace = True)

print(customers.head())

from scipy import stats
import seaborn as sns
customer_fix = pd.DataFrame()
customer_fix["Recency"] = stats.boxcox(customers["Recency"])[0]
customer_fix["Frequency"] = stats.boxcox(customers["Frequency"])[0]
customer_fix["MonetaryValue"] = pd.Series(np.cbrt(customers["MonetaryValue"])).values

print(customer_fix.tail())

fig = plt.figure(figsize=(10,5))
sns.distplot(customer_fix["Recency"], hist = False, kde = True, kde_kws = {'shade' :True, 'linewidth': 2}
, label = "Recency", color = 'green')
plt.legend(loc="upper right")
plt.show()

sns.distplot(customer_fix["Frequency"], hist = False, kde = True, kde_kws = {'shade' :True, 'linewidth': 2}
, label = "Frequency", color = 'green')
plt.legend(loc="upper right")
plt.show()

#normalizing
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
scalar.fit(customer_fix)

customer_normalized = scalar.transform(customer_fix)

print(customer_normalized.mean(axis=0).round(2))
print(customer_normalized.std(axis=0).round(2))

print(customer_normalized)

#modelling

from sklearn.cluster import KMeans
sse = {}

for k in range(1,11):
    kmeans = KMeans(n_clusters=k,random_state=42)
    kmeans.fit(customer_normalized)
    sse[k] = kmeans.inertia_

plt.title("The Elbow Method")
plt.xlabel('k')
plt.ylabel('SSE')
sns.pointplot(x=list(sse.keys()), y= list(sse.values()))
plt.show()


model = KMeans(n_clusters=3,random_state=42)
model.fit(customer_normalized)
model.labels_.shape

customers["Cluster"] = model.labels_
customers.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency' : 'mean',
    'MonetaryValue' : ['mean','count']
}).round(2)


#creating dataframe
df_normalized = pd.DataFrame(customer_normalized, columns=['Recency', 'Frequency', 'MonetaryValue'])
df_normalized['ID'] = customers.index
df_normalized['Cluster'] = model.labels_


#melting data
df_nor_melt = pd.melt(df_normalized.reset_index(), id_vars=['ID', 'Cluster'], value_vars=['Recency', 'Frequency', 'MonetaryValue'], var_name='Attribute', value_name='Value')
print(df_nor_melt.head())

#plotting

sns.lineplot(x='Attribute',y='Value', hue='Cluster', data = df_nor_melt)
plt.show()