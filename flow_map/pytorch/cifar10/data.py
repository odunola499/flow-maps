import torch
from datasets import load_dataset, Dataset as HFDataset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

def denormalize(img):
    return (img * 0.5 + 0.5).clamp(0, 1)

def preprocess(example, transform):
    example["img"] = transform(example["img"])
    return example

class Cifar10(Dataset):
    def __init__(self, data:HFDataset):

        self.transform = T.Compose(
            [T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])

        # data.set_transform(preprocess)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        img = self.transform(row['img'])

        label = row['label']
        return img, label

def get_loaders(batch_size=2, num_workers=4):
    dataset = load_dataset('uoft-cs/cifar10')
    train = Cifar10(dataset['train'])
    valid = Cifar10(dataset['test'])

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0, drop_last=True
    )
    valid_loader = DataLoader(
        valid, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )
    return train_loader, valid_loader

if __name__ == '__main__':
    from tqdm.auto import tqdm
    from torchvision.utils import save_image
    train_loader, valid_loader = get_loaders()
    for image, label in tqdm(valid_loader):
        print(image.shape)
        print(label.shape)
        save_image(denormalize(image[0]), 'test.png')
        break





