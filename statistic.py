import numpy as np
import matplotlib.pyplot as plt

def gaussian_pdf(mean, std):
    x = (np.arange(100)+0.5)/100.
    std = std+1e-6
    y =  np.exp( - (x - mean)**2 / (2 * std**2))
    return y

x = (np.arange(100)+0.5)/100.
y = gaussian_pdf(0.3, 0.03)
plt.figure()
plt.plot(x, y)
plt.savefig('./x.png')
print(x)