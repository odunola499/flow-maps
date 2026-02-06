import random
import torch
from torch.utils.data import IterableDataset, DataLoader

from flow_map.common import plot_checkerboard, checkerboard

def get_loader(
        batch_size,
        cond = False,
        max_width:int = 10,
        max_height:int = 10,
        n_samples:int = int(1e7)
):

    dataset = CheckerboardDataset(
        max_width=max_width,
        max_height=max_height,
        n_samples=n_samples, cond=cond
    )

    loader = DataLoader(
        dataset,
        collate_fn=dataset._collate_fn,
        batch_size=batch_size,
        shuffle = False,
        pin_memory = True
    )
    return loader

class CheckerboardDataset(IterableDataset):
    def __init__(self, max_width:int = 10, max_height:int = 10, n_samples:int = 1000,
                 cond = False):
        super().__init__()
        self.n_samples = n_samples
        self.max_width = max_width
        self.max_height = max_height
        self.cond = cond

    def __iter__(self):
        while True:
            #width = random.randint(1, self.max_width)
            # height = random.randint(1, self.max_height)
            if self.cond:
                width = random.randint(1, self.max_width)
                height = random.randint(1, self.max_height)
            else:
                height = self.max_height
                width = self.max_width
            board = checkerboard(
                width, height, n_samples=self.n_samples
            )
            board = torch.from_numpy(board).float()
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
    tensor_board = torch.from_numpy(board)
    rand_board = torch.randn_like(tensor_board).numpy()
    print(board.shape)
    #plot_checkerboard(rand_board)
    loader = get_loader(batch_size = 4, n_samples = 100)
    for batch in loader:
        data, width, height = batch
        print(data.shape)
        print(width.shape)
        print(height.shape)
        break
