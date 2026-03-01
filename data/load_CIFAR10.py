import torchvision.datasets as datasets
import torchvision.transforms as transforms

training_data = datasets.CIFAR10(root="data/cifar10", train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))

test_data = datasets.CIFAR10(root="data/cifar10", train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))

print(f"Training data size: {len(training_data)}")
print(f"Test data size: {len(test_data)}")