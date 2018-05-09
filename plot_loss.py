import numpy as np
import matplotlib.pyplot as plt

train_losses = np.load('all_losses.npy')

fig=plt.figure(1)
plt.plot(range(0,len(train_losses)), train_losses, 'b-')
plt.xlabel('Iterations')
plt.ylabel('Training Loss')
plt.show(True)
