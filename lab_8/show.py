import numpy as np
import matplotlib.pyplot as plt

data = np.fromfile('result.dat', dtype=float)
data = data.reshape((100, 100))

plt.figure(figsize=(8, 6))
plt.imshow(data, cmap='viridis')
plt.colorbar(label='Значения')
plt.title('Визуализация данных из result.dat')
plt.xlabel('X (NX=100)')
plt.ylabel('Y (NY=100)')

plt.savefig('plot.png', dpi=300, bbox_inches='tight')
print("График сохранён в plot.png")

plt.show()