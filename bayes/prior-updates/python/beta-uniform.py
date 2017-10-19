# plots Bayesian updates to uniform beta prior
from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import cm

x = np.linspace(0, 1, num = 100)
plt.plot(x, beta(1, 1).pdf(x), label = "Prior = beta(1,1)", color = cm.get_cmap('Blues_r')(100))
plt.plot(x, beta(1.4, 2.3).pdf(x), label = "Prior = beta(1.4,2.3)", color = cm.get_cmap('Greens_r')(100))
for i in range(1,20,2):
    plt.plot(x, beta(1+i,2).pdf(x), label = "Uniform After {} heads".format(i), cm.get_cmap('Blues_r')(100-4*i))
    plt.plot(x, beta(1.4+i,2.3).pdf(x), label = "Biased ~fter {} heads".format(i), cm.get_cmap('Greens_r')(100-4*i))

plt.legend()
plt.title("Updates to uniform prior distribution")
plt.show()