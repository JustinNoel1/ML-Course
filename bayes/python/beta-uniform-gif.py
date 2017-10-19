from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import cm
x = np.linspace(0, 1, num = 200)
# plt.plot(x, beta(1, 1).pdf(x), label = "Prior = beta(1,1)", color = cm.get_cmap('Blues_r')(100))
# plt.plot(x, beta(1.4, 2.3).pdf(x), label = "Prior = beta(1.4,2.3)", color = cm.get_cmap('Reds_r')(100))
for i in range(0,30): 
    plt.title("Posterior after seeing {} straight heads".format(i))
    plt.plot(x, beta(1+i,1).pdf(x), label = "Using Uniform Prior = Beta(1.0,1.0)".format(i), color = cm.get_cmap('Blues_r')(100))
    plt.plot(x, beta(1.4+i,2.3).pdf(x), label = "Using Prior = Beta(1.4,2.3)".format(i), color = cm.get_cmap('Reds_r')(100))
    plt.plot(x, beta(1.01+i,10.0).pdf(x), label = "Using Prior = Beta(1.01,10.0)".format(i), color = cm.get_cmap('Greys_r')(100))
    plt.ylim(0.0, 5.0)
    plt.legend(loc=2)
    plt.savefig("Beta-posterior-{:03d}".format(i))
    plt.cla()
