import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Mall_Customers.csv")

# print(df.shape)
print(df.describe())
df.drop("CustomerID",axis=1, inplace = True)
print(df.head())
print(df.isnull().sum())

plt.figure(figsize=(12,6))
plt.grid()
n=0

for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1, 3 , n)
    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
    sns.distplot(df[x], bins = 20)
    plt.title('Distplot of {}'.format(x))
plt.show()

plt.figure(figsize=(12,6))
plt.grid()
sns.countplot(y="Gender", data=df)
plt.show()


plt.figure(figsize=(12,6))

n=0
 
for cols in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1, 3, n)
    sns.set(style="whitegrid")
    plt.subplots_adjust(hspace=0.5,wspace=0.5)
    sns.violinplot(x=cols, y = 'Gender',data=df)
    plt.ylabel("Gender" if (n==1) else '')
    plt.title("Violin Plot")
plt.show()

age_18_25 = df.Age[(df.Age >= 18) & (df.Age <=25)]
age_26_35 = df.Age[(df.Age >= 26) & (df.Age <=35)]
age_36_45 = df.Age[(df.Age >= 36) & (df.Age <=45)]
age_46_55 = df.Age[(df.Age >= 46) & (df.Age <=55)]
age_55_above = df.Age[(df.Age > 55)]


agex = ['18-25','26-35','36-45','46-55','55-above']
agey = [len(age_18_25.values),len(age_26_35.values),len(age_36_45.values),len(age_46_55.values),len(age_55_above.values)]

plt.figure(figsize=(12,6))
plt.grid()
sns.barplot(x=agex, y=agey)
plt.title("Number of customer age wise")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

sns.relplot(x="Annual Income (k$)", y="Spending Score (1-100)", data=df)
plt.show()

ss_1_20 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 1) & (df["Spending Score (1-100)"] <= 20)]
ss_21_40 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 21) & (df["Spending Score (1-100)"] <= 40)]
ss_41_60 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 41) & (df["Spending Score (1-100)"] <= 60)]
ss_61_80 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 61) & (df["Spending Score (1-100)"] <= 80)]
ss_81_100 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 81) & (df["Spending Score (1-100)"] <= 100)]

ssx = ['1-20','21-40','41-60','61-80','81-100']
ssy = [len(ss_1_20.values),len(ss_21_40.values),len(ss_41_60.values),len(ss_61_80.values),len(ss_81_100.values)]

plt.figure(figsize=(12,6))
plt.grid()
sns.barplot(x=ssx,y=ssy)
plt.title("Spending Score")
plt.xlabel('Spending Score')
plt.ylabel('No of customers')
plt.show()

ai_15_30 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 15) & (df["Annual Income (k$)"] <= 30)]
ai_31_60 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 31) & (df["Annual Income (k$)"] <= 60)]
ai_61_90 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 61) & (df["Annual Income (k$)"] <= 90)]
ai_91_120 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 91) & (df["Annual Income (k$)"] <= 120)]
ai_121_137 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 121) & (df["Annual Income (k$)"] <= 137)]

xai = ['15-30','31-60','61-90','91-120','121-137']
yai = [len(ai_15_30.values),len(ai_31_60.values),len(ai_61_90.values),len(ai_91_120.values),len(ai_121_137.values)]


plt.figure(figsize=(12,6))
sns.barplot(x=xai,y=yai)
plt.title("Annual Income")
plt.xlabel("Annual Income (k$)")
plt.ylabel("No Of Customers")
plt.show()

X1 = df.loc[:,["Age", "Spending Score (1-100)"]].values

from sklearn.cluster import KMeans
wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss,linewidth=2,color="red",marker="8")
plt.xlabel("K Values")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=4)
label = kmeans.fit_predict(X1)
print(label)
print(kmeans.cluster_centers_)

plt.figure(figsize=(12,6))
plt.grid()
plt.scatter(X1[:,0],X1[:,1],c=kmeans.labels_,cmap="rainbow")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color="black")
plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.title("Cluster Of Customer")
plt.show()




X2 = df.loc[:,["Annual Income (k$)", "Spending Score (1-100)"]].values

from sklearn.cluster import KMeans
wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X2)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss,linewidth=2,color="red",marker="8")
plt.xlabel("K Values")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=5)
label = kmeans.fit_predict(X2)
print(label)
print(kmeans.cluster_centers_)



plt.figure(figsize=(12,6))
plt.grid()
plt.scatter(X2[:,0],X2[:,1],c=kmeans.labels_,cmap="rainbow")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color="black",marker="8")
plt.title("Cluster Of Customer")
plt.xlabel("Annual Income in $k")
plt.ylabel("Spending Score")
plt.show()


X3 = df.iloc[:,1:].values

from sklearn.cluster import KMeans
wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X3)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss,linewidth=2,color="red",marker="8")
plt.xlabel("K Values")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=5)
label = kmeans.fit_predict(X3)
print(label)
print(kmeans.cluster_centers_)


clusters = kmeans.fit_predict(X3)
df["label"] = clusters

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.Age[df.label==0],df["Annual Income (k$)"][df.label==0],df["Spending Score (1-100)"][df.label==0],c='blue',s=60)
ax.scatter(df.Age[df.label==1],df["Annual Income (k$)"][df.label==1],df["Spending Score (1-100)"][df.label==1],c='red',s=60)
ax.scatter(df.Age[df.label==2],df["Annual Income (k$)"][df.label==2],df["Spending Score (1-100)"][df.label==2],c='green', s=60)
ax.scatter(df.Age[df.label==3],df["Annual Income (k$)"][df.label==3],df["Spending Score (1-100)"][df.label==3],c='orange', s=60)
ax.scatter(df.Age[df.label==4],df["Annual Income (k$)"][df.label==4],df["Spending Score (1-100)"][df.label==4],c='purple', s=60)
ax.view_init(30,185)
plt.title("Customer Segmentation")
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel("Spending Score (1-100)")
plt.show()