import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.random.rand(100)
y = x * 2 + np.random.normal(0, 0.1, 100)


class SimpleLinearRegression:
    def __init__(self):
        self.x = None
        self.slope = None
        self.intercept = None
        self.covariance = None
        self.variance = None

    def fit(self, x, y):
        self.covariance = np.sum((x - np.mean(x)) * (y - np.mean(y)))
        self.variance = np.sum((x - np.mean(x)) ** 2)
        self.slope = self.covariance / self.variance
        self.intercept = np.mean(y) - self.slope * np.mean(x)

    def predict(self, x):
        return self.slope * x + self.intercept


model = SimpleLinearRegression()
model.fit(x, y)

y_pred = model.predict(x)

plt.scatter(x, y, label="Data")
plt.plot(x, y_pred, color="red", label="Fitted Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()
