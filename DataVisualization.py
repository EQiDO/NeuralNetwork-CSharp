import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_from_csv():

    train_X = np.loadtxt('x_train.csv', delimiter=',')
    train_Y = np.loadtxt('y_train.csv', delimiter=',')

    x, y, z = train_X
    labels = train_Y.flatten()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x[labels==0], y[labels==0], z[labels==0], c='blue', label='f < 0')
    ax.scatter(x[labels==1], y[labels==1], z[labels==1], c='red', label='f > 0')

    dummy_x = np.linspace(-10, 20, 80)
    dummy_y = np.linspace(-10, 20, 80)
    f_X, f_Y = np.meshgrid(dummy_x, dummy_y)

    f_Z = 8 - (f_X - 3)**2 - (f_Y - 5)**2
    f_Z = np.clip(f_Z, -10, 20)

    ax.plot_surface(f_X, f_Y, f_Z, alpha=0.3, color='gray', edgecolor='none')

    ax.set_zlim(-10, 20)
    ax.set_xlim(-10, 20)
    ax.set_ylim(-10, 20)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Scatter Plot of Training Data")
    ax.legend()

    plt.show()

visualize_from_csv()