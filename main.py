import pandas
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
data = pandas.read_csv('apple_stock.csv')
model = LinearRegression()
model.fit(data[['year']],data[['price']])
print(model.predict([[100]]))
plt.scatter(data['year'],data['price'])
plt.show()



