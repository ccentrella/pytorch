from model.ModelWrapper import ModelWrapper
from model.NeuralNetwork import NeuralNetwork
from model.FashionMNISTData import FashionMNISTData
from Device import Device

# Get output device
print()
Device.print_device()
device = Device.get_cpu()

# Get training and test data
data = FashionMNISTData()
data.print_data_shape()
train_dataloader, test_dataloader = data.get_data()

# Load and train model
model = NeuralNetwork(device).to(device)
wrapper = ModelWrapper(train_dataloader, test_dataloader, model)
wrapper.train()
wrapper.save()