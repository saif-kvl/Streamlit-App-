import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

dataset = pd.read_csv(r"C:\Users\SAIF SHAIK\Downloads\Salary_Data.csv")

x = dataset.iloc[:,:-1]

y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split

x_train ,x_test ,y_train,y_test = train_test_split(x,y,test_size=0.2)


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

comparison = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(comparison)

plt.scatter(x_test, y_test ,color='red' )
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years of Experience')
plt.ylabel("Salary")
plt.show()

m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

y_12 = m_slope*12 + c_intercept
print(y_12)

bias = regressor.score(x_train,y_train)
print(bias)

variance = regressor.score(x_test,y_test)
print(variance)

#Statistics concepts need to add 

dataset.mean()

dataset.median()

dataset['Salary'].mode()

dataset['Salary'].var()

dataset.std()

dataset['Salary'].std()

from scipy.stats import variation

variation(dataset.values)

variation(dataset['Salary'])
#Correlation

dataset.corr()

dataset['Salary'].corr(dataset['YearsExperience'])

#Skewness 

dataset.skew()

dataset['Salary'].skew()

#Standard Error

dataset.sem()

dataset['Salary'].sem()

#Z-Score

import scipy.stats as stats

dataset.apply(stats.zscore)

stats.zscore(dataset)



#Degree of freedom

a = dataset.shape[0]
b = dataset.shape[1]

degree_of_freedom = a-b
print(degree_of_freedom)



#sum of square regresso (ssr)

y_mean = np.mean(y)
ssr = np.sum((y_pred-y_mean)**2)
print(ssr)

#sse
y = y[0:6]
sse = np.sum((y-y_pred)**2)
print(sse)

#sst
mean_total= np.mean(dataset.values)
sst = np.sum((dataset.values-mean_total)**2)
print(sst)


#R-Square 

r_Square = 1-(ssr/sst)
r_Square


import pickle
filename = 'linear_regression_model.pkl'
with open(filename ,'wb') as file:
    pickle.dump(regressor,file)
print("Model has been pickled and saved as linear_regression_model.pkl")

import os 
print(os.getcwd())




import streamlit as st
import pickle
import numpy as np
# Load the pre-trained model

model = pickle.load(open(r'C:\Users\SAIF SHAIK\OneDrive\Desktop\FSDS PRACTICES FILES ALL\linear_regression_model.pkl', 'rb'))
st.title("Salary Prediction Based on Years of Experience")
st.write("This application predicts the salary based on the years of experience using a linear regression model.")
st.write("Enter the number of years of experience to get the predicted salary.")
# Input for years of experience

years_of_experience = st.number_input("Enter Years of Experience",min_value=0, max_value=50, value=0)

# when the button is clicked, make the prediction

if st.button("Predict Salary"):
# Make prediction using the modelprint()
    experience_input = np.array([[years_of_experience]])
# Predict the salary
    prediction = model.predict(experience_input)
# Display the predicted salary

    st.success(f"Predicted Salary for {years_of_experience} years of experience is: ${prediction[0]:,.2f}")

# Display the predicted salary in a more detailed format
# Show detailed prediction only if prediction is made
if 'prediction' in locals():
    st.write(f"Predicted Salary: ${prediction[0]:,.2f}")
    
    





