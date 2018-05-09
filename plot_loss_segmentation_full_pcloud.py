import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--file', type=str, default = '',  help='model path')

opt = parser.parse_args()

loss_train = np.load(opt.file)['loss_train']
plot_train = plt.plot((loss_train), 'b-*', label='train')
plt.grid()
plt.show(True)