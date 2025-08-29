import intelligen
import numpy as np
import matplotlib.pyplot as plt

print(intelligen.special.erf(1.0 + 1j))
x = np.linspace(-5, 5, 400)
y = intelligen.special.erf(x)
plt.plot(x, y)
plt.title('Error Function')
plt.xlabel('x')
plt.ylabel('erf(x)')
plt.grid()
plt.show()
