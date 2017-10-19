from pymc3 import *
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures 
import pandas as pd
import theano
from scipy.stats.kde import gaussian_kde
NUM_SAMPLES = 30
SDEV = 0.3
np.random.seed(42)

def f(x): 
    return 7 *(x**3 -1.3 *x**2+0.5*x - 0.056)

gridx = np.linspace(0,1,200)
x = np.random.random(NUM_SAMPLES)
truey = f(x) 
y = truey + SDEV*np.random.randn(NUM_SAMPLES)

# poly = PolynomialFeatures(degree = 10)
# xpow = poly.fit_transform(x.reshape(NUM_SAMPLES,1))
# linpow = poly.fit_transform(gridx.reshape(200,1))
data = dict(x1=theano.shared(x), y=theano.shared(y))
print(data['x1'][:5])
fig = plt.figure()
ax = fig.add_subplot(111, title = 'Bayesian approach to degree 10 polynomial regression')
ax.scatter(x,y, label = 'Sampled data')
ax.plot(gridx, f(gridx), label = 'True curve', linewidth = 4 )

with Model() as model:
    # glm.GLM.from_formula('y ~ x1 + x2 + x3 ', data)
    coeff = Normal('coeff', mu = 0, sd = 10, shape = 11)
    sigma = HalfNormal('sigma', sd = 1)
    # mu = coeff[0] + coeff[1]*data['x1'] + coeff[2]*data['x2']+ coeff[3]*data['x3']
    mu = np.sum([coeff[i]*x**i for i in range(0,11)])
    y_obs = Normal('y_obs', mu = mu, sd = sigma, observed = data['y'])
    trace = sample(15000, njobs = 4, tune = 1000 )
    # print(trace['coeff'])
    print(stats.df_summary(trace).to_html())
    mcoeff = np.mean(trace['coeff'], axis = 0)
    def g(x):
        return np.sum([mcoeff[i]*x**i for i in range(11)], axis =0)
    ax.plot(gridx, g(gridx), label = 'Bayesian mean polynomial fit', color = 'Black', alpha = 0.5, lw = 2.5)
    ax.set_ylim(-1., 1.25)

    for i in range(50):
        def h(x):
            return np.sum([trace['coeff'][-i][j]*x**j for j in range(4)], axis =0)
        ax.plot(gridx, h(gridx), color = 'Black', alpha = 0.1)
    gridx = np.linspace(0,1,30)
    data['x1'].set_value(gridx)
    # data['x2'].set_value(linpow[:,2])
    # data['x3'].set_value(linpow[:,3])
    data['y'].set_value(np.zeros_like(gridx))
    post_pred = sample_ppc(trace, samples = 200)
    print(post_pred['y_obs'].shape)

    ax.plot(gridx, np.mean(post_pred['y_obs'], axis = 0), label = 'Bayesian mean posterior', alpha = 0.5, color = 'Red')
    ax.fill_between(gridx, np.mean(post_pred['y_obs'], axis = 0)-np.std(post_pred['y_obs'], axis = 0), np.mean(post_pred['y_obs'], axis = 0)+np.std(post_pred['y_obs'], axis = 0), label = 'Bayesian error band', alpha = 0.1, color = 'Red')
    plt.legend(loc = 2)
    plt.savefig("bayreg10.png")
    plt.show()
    plt.clf()
    plt.close()
    plt.figure(figsize = (5, 24))
    fig, axs = plt.subplots(12)
    for i in range(11):
        tdata = trace['coeff'][:,i]
        kde = gaussian_kde(tdata)
        tgridx = np.linspace(min(tdata), max(tdata), 200)
        axs[i].plot(tgridx, kde(tgridx))
        axs[i].set_ylim(axs[i])
        axs[i].set_title('Coeff{}'.format(i))

    tdata = trace['sigma']
    kde = gaussian_kde(tdata)
    tgridx = np.linspace(min(tdata), max(tdata), 200)
    axs[11].plot(tgridx, kde(tgridx)) 
    axs[11].set_ylim(axs[11])
    axs[11].set_title('Sigma')
    plt.savefig("trace10.png")


    # plot_posterior_predictive_glm(trace, samples = 200, label ='Posterior predictive regression')
    # traceplot(trace)
    # plt.tight_layout()


