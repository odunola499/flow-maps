import numpy as np
import matplotlib.pyplot as plt

def checkerboard(nx, ny, n_samples):
    total_samples = 0
    samples = np.array([]).reshape((0,2))
    while total_samples < n_samples:
        curr_samples = np.random.rand(
            n_samples * 2, 2
        )

        x_idx = (curr_samples[:,0] * nx).astype(int)
        y_idx = (curr_samples[:,1] * ny).astype(int)

        mask = (x_idx + y_idx) % 2 == 0
        curr_samples = curr_samples[mask]

        samples = np.concatenate((samples, curr_samples))
        total_samples  = samples.shape[0]

    return (2 * samples[:n_samples] - 1).astype(np.float32)

def plot_checkerboard(x, title="Checkerboard", save:str = None):
    plt.figure(figsize=(5, 5))
    plt.scatter(x[:, 0], x[:, 1], s=2)
    plt.gca().set_aspect('equal')
    plt.title(title)
    if save:
        plt.savefig(save, dpi = 300)
    else:
        plt.show()