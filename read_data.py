import pandas as pd
#1. Loading dataset
# Load dataset (adjust file name if it's .xlsx)
df = pd.read_csv("C:\\Users\\chand\\OneDrive\\Desktop\\indian-cities-dataset.csv")

# Display first few rows
print(df.head())



#2. Preprocessing
print(df.size)
print(df.shape)
df.info()
print(df.isnull().sum())



#3.visualisation
import matplotlib.pyplot as plt
import seaborn as sns

#Visualization 2: Box Plot of Distances by Origin
plt.figure(figsize=(20, 10))
sns.boxplot(data=df, x='Origin', y='Distance', hue='Origin', palette='viridis')  # ✅ Fixed
plt.title('Box Plot of Distances by origin')
plt.xlabel('origin')
plt.ylabel('Distance ')
plt.xticks(rotation=45)
plt.legend(title='Origin')
plt.show()

#Visualization 3: Scatter Plot of Distances by origin and Destination

plt.figure(figsize=(20, 16))
sns.boxplot(data=df, x='Origin', y='Distance')

plt.title('Scatter Plot of Distances by Origin and Destination')
plt.xlabel('Origin')
plt.ylabel('Destination')
plt.legend(title='Distance', bbox_to_anchor=(1, 1))
plt.show()



#4. Divide input and output
x=df.iloc[:,0:2].values
print(x)

y=df.iloc[:,2].values
print(y)

df['Distance']=df['Distance'].astype(float)
print(df.dtypes)



#5. train and test the variable
from sklearn.model_selection import train_test_split

#library-train_test_split, sklearn.model_selection is package
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)

print(x.shape)#100% data
print(x_train.shape) #75% data
print(x_test.shape) #25% data

print(y.shape)#100% data
print(y_train.shape)#75% data
print(y_test.shape)#25% data



#6. Normalisation

from sklearn.preprocessing import MinMaxScaler

#Assuming 'distance_km' is the feature you want to normalize
feature_to_normalize=df[['Distance']]

#Initialize the MinMaxScaler 
scaler = MinMaxScaler()
# Fit and transform the feature using Min-Max scaling
normalized_feature = scaler.fit_transform(feature_to_normalize)

#Replace the original column with the normalized values
df['normalized_distance'] = normalized_feature

#Display the normalized dataset
print(df.head())



#7. rum the regressor, classsifier or clusterer

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.preprocessing import StandardScaler

#Assuming 'featurel', 'feature2',... are your input features and 'distance_km' is the target variable
x = df[['Origin', 'Destination']] #Adjust based on your dataset
y=df['Distance'] #Assuming 'distance_ku' is your target variable

#Convert string features to float using one-hot encoding
x_encoded = pd.get_dummies(x, columns=['Origin', 'Destination'])

#Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.2, random_state=42)

#Standardize features (optional but can be beneficial for linear regression)
scaler=StandardScaler()
X_train_scaled =scaler.fit_transform(x_train)
X_test_scaled =scaler.transform(x_test)

#Initialize the Linear Regression model
model =LinearRegression()



#8. Fit the model 
#Train the model on the training set
model.fit(x_train, y_train)


#9. Predict the output
y_pred=model.predict(x_test) #using input test data, we predict
print(y_pred) # predicted output values

y_test


#conclusion
# we have to compare corresponding values of y_pred and y
#50 when we compare, we come to know that there is huge difference
#this huge difference does not mean, our model has predicted wrong
#it only means our model is not linear or lesslinear
#linearity of a model depends on nature of the data
#as well as size of the data


#Tfidf vectorisation
#individual prediction
a=df['Origin'][10]
print(a)

b=df['Destination'][10]
print(b)


