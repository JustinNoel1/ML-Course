from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import cm
from sklearn.metrics import mean_squared_error

NUM_SAMPLES = 100
np.random.seed(42)

def f(x): 
    return 7 *(x**3 -1.3 *x**2+0.5*x - 0.056)
data = np.array([[x,f(x) ] for x in np.random.random(NUM_SAMPLES)])
orig_data = np.copy(data)
# data[:,1] += 0.3*np.random.randn(NUM_SAMPLES)
gridx = np.linspace(0, 1, NUM_SAMPLES)
gridy = np.array([f(x) for x in gridx])

datax = data[:,0]
normaly = data[:,1]+0.3*np.random.randn(NUM_SAMPLES)

plt.scatter(data[:,0], normaly )
plt.title("Scatter plot of synthetic data with normal errors")
plt.plot(gridx, gridy, label = "True function", color = 'Red')
plt.legend(loc = 2)
plt.savefig("poly_scatter_normal.png")
plt.cla()

# laplacey = data[:,1]+np.random.laplace(scale = 0.3/np.sqrt(2), size = NUM_SAMPLES)
# plt.scatter(data[:,0], laplacey)
# plt.title("Scatter plot of synthetic data with Laplacian errors")
# plt.plot(gridx, gridy, label = "True function", color = 'Red')
# plt.legend(loc = 2)
# plt.savefig("scatter_laplace.png")
# plt.cla()

gen_poly = True
if gen_poly:
    lm = LinearRegression()
    for deg in range(1, 8):
        poly = PolynomialFeatures(degree = deg)
        newdatax = poly.fit_transform(datax.reshape(NUM_SAMPLES,1))
        for i in range(1, NUM_SAMPLES+1):
            lm.fit(newdatax[:i], normaly[:i].reshape(i, 1))
            predictions = lm.predict(newdatax)
            # print(predictions)
            mse = mean_squared_error(predictions, normaly.reshape(NUM_SAMPLES,1))
            plt.ylim(-0.75, 1.25)
            plt.scatter(datax, normaly)
            plt.title("Degree {} polynomial regression on {} points with normal error".format(deg, i))
            plt.plot(gridx, gridy, label = "True function", color = 'Red')
            gridpred = lm.predict(poly.fit_transform(gridx.reshape(NUM_SAMPLES, 1)))
            plt.plot(gridx.flatten(), gridpred.flatten(), label = "Polynomial regressor curve MSE = {:0.4f}".format(mse), color = 'Green')
            plt.legend(loc = 2)
            plt.savefig("polyreg_normal_{:02d}{:03d}.png".format(deg,i))
            plt.cla()   

gen_var = False
if gen_var:
    lm = LinearRegression()
    poly = PolynomialFeatures(degree = 10)
    newdatax = poly.fit_transform(datax.reshape(NUM_SAMPLES,1))
    for i in range(30):
        samp = np.random.choice(range(NUM_SAMPLES), 30)
        lm.fit(newdatax[samp], normaly[samp].reshape(30, 1))
        predictions = lm.predict(newdatax)
        # print(predictions)
        mse = mean_squared_error(predictions, normaly.reshape(NUM_SAMPLES,1))
        plt.ylim(-0.75, 1.25)
        plt.scatter(datax, normaly)
        plt.title("Degree {} polynomial regression on 30 random points with normal error".format(10))
        plt.plot(gridx, gridy, label = "True function", color = 'Red')
        gridpred = lm.predict(poly.fit_transform(gridx.reshape(NUM_SAMPLES, 1)))
        plt.plot(gridx.flatten(), gridpred.flatten(), label = "Polynomial regressor curve MSE = {:0.4f}".format(mse), color = 'Green')
        plt.legend(loc = 2)
        plt.savefig("polyreg_var_{:03d}.png".format(i))
        plt.cla()   
# lm = LinearRegression()
# for i in range(1, NUM_SAMPLES+1):
#     lm.fit(datax[:i].reshape((i,1)), laplacey[:i].reshape((i,1)))
#     predictions = lm.predict(datax.reshape(NUM_SAMPLES,1))
#     mse = mean_squared_error(predictions, laplacey)
#     plt.scatter(datax, normaly)
#     plt.title("Linear regression on {} points with Laplacian error".format(i))
#     plt.plot(gridx, gridy, label = "True function", color = 'Red')
#     plt.plot(gridx, [lm.coef_[0] * x + lm.intercept_[0]  for x in gridx], label = "Linear regressor line MSE = {:0.4f}".format(mse), color = 'Green')
#     plt.legend(loc = 2)
#     plt.savefig("linreg_laplace_{:03d}.png".format(i))
#     plt.cla()   
