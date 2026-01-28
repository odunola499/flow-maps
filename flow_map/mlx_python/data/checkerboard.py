import random
import mlx.core as mx

from flow_map.common import plot_checkerboard, checkerboard

def checkerboard_generator(
        batch_size,
        n_samples = 1000,
        max_width = 10,
        max_height = 10
):
    while True:
        batch_data = []
        widths = []
        heights = []

        for _ in range(batch_size):
            width = random.randint(1, max_width)
            height = random.randint(1, max_height)
            board = checkerboard(width, height, n_samples)
            board = mx.array(board)
            batch_data.append(board)
            widths.append(width)
            heights.append(height)

        batch_data = mx.stack(batch_data)
        widths = mx.array(widths)
        heights = mx.array(heights)

        yield batch_data, widths, heights

def get_loader(
        batch_size,
        max_width = 10,
        max_height = 10,
        n_samples = 1000,
):

    loader = checkerboard_generator(
        batch_size = batch_size,
        n_samples = n_samples,
        max_width = max_width,
        max_height = max_height
    )
    return loader

if __name__ == '__main__':

    board = checkerboard(4, 4, 1000)
    plot_checkerboard(board)
    loader = get_loader(batch_size = 4, n_samples = 100)
    for batch in loader:
        data, width, height = batch
        print(data.shape)
        print(width.shape)
        print(height.shape)
        break

