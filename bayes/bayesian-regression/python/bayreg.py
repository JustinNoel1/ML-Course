#Implementation of Bayesian polynomial regression using pymc3
from pymc3 import *
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures 
import pandas as pd
import theano
from scipy.stats.kde import gaussian_kde

# set sample size
NUM_SAMPLES = 200
# set desired standard error
SDEV = 0.3
# set random seed
np.random.seed(42)

# set true function
def f(x): 
    return 7 *(x**3 -1.3 *x**2+0.5*x - 0.056)

# evenly spaced grid of points in unit interval
gridx = np.linspace(0,1, 200)
# sample from unit interval
x = np.random.random(NUM_SAMPLES)
# perturb values
y = f(x) + SDEV*np.random.randn(NUM_SAMPLES)

# construct polynomial features on the samples and the grid
poly = PolynomialFeatures(degree = 3)
xpow = poly.fit_transform(x.reshape(NUM_SAMPLES,1))
linpow = poly.fit_transform(gridx.reshape(200,1))

# Create dictionary of theano variables
data = dict(x1=theano.shared(xpow[:,1]), x2 = theano.shared(xpow[:,2]), x3 = theano.shared(xpow[:,3]), y=theano.shared(y))

# plot our true curve
fig = plt.figure()
ax = fig.add_subplot(111, title = 'Bayesian approach to polynomial regression')
ax.scatter(x,y, label = 'Sampled data')
ax.plot(gridx, f(gridx), label = 'True curve', linewidth = 4 )

# this is needed to work in pymc3 context
with Model() as model:
    # Build the pymc3 model
    coeff = Normal('coeff', mu = 0, sd = 10, shape = 4)
    sigma = HalfNormal('sigma', sd = 1)
    mu = np.sum([coeff[i]*data['x'+str(i)] for i in range(1,4)])+coeff[0]
    y_obs = Normal('y_obs', mu = mu, sd = sigma, observed = data['y'])

    # Calculate the posterior probability via MCMC integration
    trace = sample(3500, njobs = 4)

    # print summary of training in html format
    print(stats.df_summary(trace).to_html())

    # Get the mean coefficients
    mcoeff = np.mean(trace['coeff'], axis = 0)
    # define the predicted function using the mean polynomial
    def g(x):
        return np.sum([mcoeff[i]*x**i for i in range(4)], axis =0)

    # plot the mean polynomial
    ax.plot(gridx, g(gridx), label = 'Bayesian mean polynomial fit', color = 'Black', alpha = 0.5, lw = 2.5)

    # take the last 50 samples of the parameters and graph their associated polynomials
    for i in range(50):
        def h(x):
            return np.sum([trace['coeff'][-i][j]*x**j for j in range(4)], axis =0)
        ax.plot(gridx, h(gridx), color = 'Black', alpha = 0.1)

    # Calculate posterior values
    data['x1'].set_value(linpow[:,1])
    data['x2'].set_value(linpow[:,2])
    data['x3'].set_value(linpow[:,3])
    data['y'].set_value(np.zeros_like(linpow[:1,1]))
    post_pred = sample_ppc(trace, samples = 200)

    # plot the mean prediction plus +- one standard deviation 
    ax.plot(gridx, np.mean(post_pred['y_obs'], axis = 0), label = 'Bayesian mean posterior', alpha = 0.5, color = 'Red')
    ax.fill_between(gridx, np.mean(post_pred['y_obs'], axis = 0)-np.std(post_pred['y_obs'], axis = 0), np.mean(post_pred['y_obs'], axis = 0)+np.std(post_pred['y_obs'], axis = 0), label = 'Bayesian error band', alpha = 0.1, color = 'Red')
    plt.legend(loc = 2)
    plt.savefig("bayreg.png")
    plt.show()
    plt.clf()
    plt.close()

    # plot the pdfs of the coefficients
    plt.figure(figsize = (10, 5))
    fig, axs = plt.subplots(5)
    for i in range(4):
        tdata = trace['coeff'][:,i]
        kde = gaussian_kde(tdata)
        tgridx = np.linspace(min(tdata), max(tdata), 200)
        axs[i].plot(tgridx, kde(tgridx))
        axs[i].set_title('Coeff{}'.format(i))

    tdata = trace['sigma']
    kde = gaussian_kde(tdata)
    tgridx = np.linspace(min(tdata), max(tdata), 200)
    axs[4].plot(tgridx, kde(tgridx)) 
    axs[4].set_title('Sigma')
    plt.tight_layout()
    plt.savefig("trace.png") 
    plt.show()


