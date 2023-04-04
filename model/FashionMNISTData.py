from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class FashionMNISTData:

    def __init__(self) -> None:
        train_data, test_data = self.__fetch_data()
        batch_size = 64

        self.train_dataloader = DataLoader(train_data, batch_size=batch_size)
        self.test_dataloader = DataLoader(test_data, batch_size=batch_size)

    def get_data(self) -> Tuple[object, object]:
        return (self.train_dataloader, self.test_dataloader)
    
    def print_data_shape(self) -> None:
        for X, y in self.test_dataloader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}\n")
            break
    
    def __fetch_data(self) -> Tuple[object, object]:
        train_data = datasets.FashionMNIST(root='data',
                                        train=True,
                                        download=True,
                                        transform=ToTensor(),
                                        )
        test_data = datasets.FashionMNIST(root='data',
                                        train=False,
                                        download=True,
                                        transform=ToTensor(),
                                        )
        return (train_data, test_data)
    