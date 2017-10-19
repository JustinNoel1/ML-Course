from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import cm
from sklearn.metrics import mean_squared_error

# Fix the number of samples and our seed
NUM_SAMPLES = 200
np.random.seed(42)

# Our "true function"
def f(x): 
    return 1.5*x + 0.5

#Construct array of (x,f(x))-pairs where x is sampled randomly from unit interval
data = np.array([[x,f(x) ] for x in np.random.random(NUM_SAMPLES)])

# Create regular grid of x values and the values of f
gridx = np.linspace(0, 1, 200)
gridy = np.array([f(x) for x in gridx])

# Add Gaussian noise with sigma=0.6
normaly = data[:,1]+0.6*np.random.randn(NUM_SAMPLES)

#Plot the messy data
plt.scatter(data[:,0], normaly )
plt.title("Scatter plot of synthetic data with normal errors")
#Plot the true function
plt.plot(gridx, gridy, label = "True function", color = 'Red')
plt.legend(loc = 2)
# Save and clear
plt.savefig("scatter_normal.png")
plt.cla()


# Fit linear regressors to increasingly large intervals of data
lm = LinearRegression()
for i in range(1, NUM_SAMPLES+1):
    # Fit the regressor
    lm.fit(data[:i,0].reshape((i,1)), normaly[:i].reshape((i,1)))
    # Get the predictions on all of the sample points
    predictions = lm.predict(data[:,0].reshape(NUM_SAMPLES,1))

    # Get MSE
    mse = mean_squared_error(predictions, normaly)

    # Plot the messy data
    plt.scatter(data[:,0], normaly)
    plt.title("Linear regression on {} points with normal error".format(i))
    # Plot the true function
    plt.plot(gridx, gridy, label = "True function", color = 'Red')
    # Plot the regression line
    plt.plot(gridx, [lm.coef_[0] * x + lm.intercept_[0]  for x in gridx], label = "Linear regressor line MSE = {:0.4f}".format(mse), color = 'Green')
    plt.legend(loc = 2)
    # Save and clear
    plt.savefig("linreg_normal_{:03d}.png".format(i))
    plt.cla()   
