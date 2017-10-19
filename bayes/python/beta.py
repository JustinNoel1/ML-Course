from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt 
x = np.linspace(0, 1, num = 100)
plt.plot(x, beta(1.4, 2.3).pdf(x), label = "Prior = beta(1.4,2.3)")
for i in range(1,11,2):
    plt.plot(x, beta(1.4+i,2.3).pdf(x), label = "After {} heads".format(i))
for i in range(1,6,2):
    plt.plot(x, beta(1.4+i*5+10,2.3).pdf(x), label = "After {} heads".format(i*5+10))   
plt.legend()
plt.title("Updates to prior distribution")
plt.show()