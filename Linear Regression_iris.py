import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

iris=pd.read_csv('Iris.csv')
#print(iris.head())
#print(iris[iris["SepalWidthCm"]>4])
#print(iris[iris['PetalWidthCm']>1])
#print(iris[iris['PetalWidthCm']>2])
#sns.scatterplot(x='sepal_length',y='petal_length',data=iris,hue='species')
#plt.show()

#Variability

#MODEL 1

#x=iris[['sepal_width']]
#y=iris[['sepal_length']]
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
#print(x_train.head())
#print(x_test.head())
#print(y_train.head())
#print(y_test.head())
#lr=LinearRegression()
#lr.fit(x_train,y_train)
#y_predicted=lr.predict(x_test)
#print(y_test.head())
#print(y_predicted[0:5])
#print(mean_squared_error(y_test,y_predicted))

#MODEL 2(MULTIPLE REGRESSION-MORE INDEPENDENT VARIABLES
#                            MORE IS THE ACCURACY OF THE MODEL

x=iris[['sepal_width','petal_length','petal_width']]
y=iris[['sepal_length']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
lr2=LinearRegression()
lr2.fit(x_train,y_train)
y_predicted=lr2.predict(x_test)
print(mean_squared_error(y_test,y_predicted))