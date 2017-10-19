from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import cm
x = np.linspace(0, 1, num = 100)
plt.plot(x, beta(1, 1).pdf(x), label = "Prior = beta(1,1)", color = cm.get_cmap('Blues_r')(100))
plt.plot(x, beta(1.4, 2.3).pdf(x), label = "Prior = beta(1.4,2.3)", color = cm.get_cmap('Reds_r')(100))
for i in range(1,22,5):
    plt.plot(x, beta(1+i,2).pdf(x), label = "Uniform After {} heads".format(i), color = cm.get_cmap('Blues_r')(100-10*i))
    plt.plot(x, beta(1.4+i,2.3).pdf(x), label = "Biased After {} heads".format(i), color = cm.get_cmap('Reds_r')(100-10*i))

plt.legend()
plt.title("Comparison of updates to uniform and non-uniform priors")
plt.show()