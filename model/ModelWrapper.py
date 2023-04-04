import os.path
import torch
from torch import nn
from Device import Device

class ModelWrapper:

    def __init__(self, train_dataloader, test_dataloader, model) -> None:
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model

        self.__load()

    def train(self) -> None:
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        epochs = 5

        print("Starting training...\n")
        for t in range(epochs):
            print(f'Epoch {t+1}\n-------------------------------------')
            self.__train_epoch(self.train_dataloader, self.model, loss_fn, optimizer)
            self.__test_epoch(self.test_dataloader, self.model, loss_fn)
        print('Done!\n')
    
    def save(self) -> None:
        print("Saving model...\n")
        torch.save(self.model.state_dict(), "model.pth")
        print("Model saved successfully.\n")

    def __load(self) -> None:
        if os.path.exists("model.pth"):
            self.model.load_state_dict(torch.load("model.pth"))
            print(f'{self.model}\n')

    def __train_epoch(self, dataloader, model, loss_fn, optimizer) -> None:
        device = Device.get_cpu()
        size = len(dataloader.dataset)
        model.train()

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

        print()

    def __test_epoch(self, dataloader, model, loss_fn) -> None:
        device = Device.get_cpu()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()

        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f'Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')