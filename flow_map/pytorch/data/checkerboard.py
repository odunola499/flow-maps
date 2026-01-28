import random
import torch
from torch.utils.data import IterableDataset, DataLoader

from flow_map.common import plot_checkerboard, checkerboard

def get_loader(
        batch_size,
        max_width:int = 10,
        max_height:int = 10,
        n_samples:int = int(1e7)
):

    dataset = CheckerboardDataset(
        max_width=max_width,
        max_height=max_height,
        n_samples=n_samples
    )

    loader = DataLoader(
        dataset,
        collate_fn=dataset._collate_fn,
        batch_size=batch_size,
        shuffle = False
    )
    return loader

class CheckerboardDataset(IterableDataset):
    def __init__(self, max_width:int = 10, max_height:int = 10, n_samples:int = 1000):
        super().__init__()
        self.n_samples = n_samples
        self.max_width = max_width
        self.max_height = max_height

    def __iter__(self):
        while True:
            #width = random.randint(1, self.max_width)
            #height = random.randint(1, self.max_height)
            width = self.max_width
            height = self.max_height
            board = checkerboard(
                self.max_width, self.max_height, n_samples=self.n_samples
            )
            board = torch.from_numpy(board)
            yield board, width, height

    def _collate_fn(self, batch):
        data = [i[0] for i in batch]
        width = [i[1] for i in batch]
        height = [i[2] for i in batch]

        data = torch.stack(data)
        width = torch.tensor(width, dtype = torch.long)
        height = torch.tensor(height, dtype = torch.long)
        return data, width, height



if __name__ == "__main__":
    board = checkerboard(6, 4, 1000)
    print(board.shape)
    plot_checkerboard(board)
    loader = get_loader(batch_size = 4, n_samples = 100)
    for batch in loader:
        data, width, height = batch
        print(data.shape)
        print(width.shape)
        print(height.shape)
        break
