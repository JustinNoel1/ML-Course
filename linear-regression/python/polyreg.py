# This file implements polynomial regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import cm
from sklearn.metrics import mean_squared_error

#Set number of samples and seed
NUM_SAMPLES = 100
np.random.seed(42)

# Our `True' function
def f(x): 
    return 7 *(x**3 -1.3 *x**2+0.5*x - 0.056)

# initialize sample data
data = np.array([[x,f(x) ] for x in np.random.random(NUM_SAMPLES)])

# grid of coordinates for true function
gridx = np.linspace(0, 1, NUM_SAMPLES)
gridy = np.array([f(x) for x in gridx])

datax = data[:,0]
normaly = data[:,1]+0.3*np.random.randn(NUM_SAMPLES)

#Plot sampled data points
plt.scatter(datax, normaly )
plt.title("Scatter plot of synthetic data with normal errors")
plt.plot(gridx, gridy, label = "True function", color = 'Red')
plt.legend(loc = 2)
plt.savefig("poly_scatter_normal.png")
plt.cla()

gen_poly = True
# Run polynomial regression repeatedly for increasing degrees
if gen_poly:
    lm = LinearRegression()
    for deg in range(1, 8):
        poly = PolynomialFeatures(degree = deg)
        newdatax = poly.fit_transform(datax.reshape(NUM_SAMPLES,1))
        for i in range(1, NUM_SAMPLES+1):
            lm.fit(newdatax[:i], normaly[:i].reshape(i, 1))
            predictions = lm.predict(newdatax)
            mse = mean_squared_error(predictions, normaly.reshape(NUM_SAMPLES,1))

            #Plot everything
            plt.ylim(-0.75, 1.25)
            plt.scatter(datax, normaly)
            plt.title("Degree {} polynomial regression on {} points with normal error".format(deg, i))
            plt.plot(gridx, gridy, label = "True function", color = 'Red')
            gridpred = lm.predict(poly.fit_transform(gridx.reshape(NUM_SAMPLES, 1)))
            plt.plot(gridx.flatten(), gridpred.flatten(), label = "Polynomial regressor curve MSE = {:0.4f}".format(mse), color = 'Green')
            plt.legend(loc = 2)
            plt.savefig("polyreg_normal_{:02d}{:03d}.png".format(deg,i))
            plt.cla()   

# Run degree 10 polynomial regression repeatedly using a random sample of 30 points
gen_var = True
if gen_var:
    lm = LinearRegression()
    poly = PolynomialFeatures(degree = 10)
    newdatax = poly.fit_transform(datax.reshape(NUM_SAMPLES,1))
    for i in range(30):
        samp = np.random.choice(range(NUM_SAMPLES), 30)
        lm.fit(newdatax[samp], normaly[samp].reshape(30, 1))
        predictions = lm.predict(newdatax)
        mse = mean_squared_error(predictions, normaly.reshape(NUM_SAMPLES,1))

        #Plot everything 
        plt.ylim(-0.75, 1.25)
        plt.scatter(datax, normaly)
        plt.title("Degree {} polynomial regression on 30 random points with normal error".format(10))
        plt.plot(gridx, gridy, label = "True function", color = 'Red')
        gridpred = lm.predict(poly.fit_transform(gridx.reshape(NUM_SAMPLES, 1)))
        plt.plot(gridx.flatten(), gridpred.flatten(), label = "Polynomial regressor curve MSE = {:0.4f}".format(mse), color = 'Green')
        plt.legend(loc = 2)
        plt.savefig("polyreg_var_{:03d}.png".format(i))
        plt.cla()   
