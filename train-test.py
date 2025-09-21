import torch
import torch.optim as optim
import torch.nn as nn

def load_dataset(batch_size, augmentation=False, shuffle=False):
  transform = Compose([
      ToTensor(),
      RandomHorizontalFlip(p=0.3),
      RandomRotation(degrees=15)
  ]) if augmentation else ToTensor()
  
  training_data = datasets.FashionMNIST(
      root="data",
      train=True,
      download=True,
      transform=transform
  )

  test_data = datasets.FashionMNIST(
      root="data",
      train=False,
      download=True,
      transform=ToTensor()
  )

  train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle)
  test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)
  return train_dataloader, test_dataloader


def train_loop(dataloader, model, loss_fn, optimizer, log_every_n_batches=100):
    # Number of dataset elements
    model.train()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Log Training Results every number of batches
        if batch % log_every_n_batches == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}, current batch: ({batch+1}/{num_batches})")


def accuracy(dataloader, model, loss_fn):
  with torch.no_grad():
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    test_loss, correct = 0, 0
    for X, y in dataloader:
        pred = model(X)
        test_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")
    return correct, test_loss


def run_train_model(model, batch_size=128, init_learning_rate=0.1, epochs=30, momentum=0.9, augmentation=False, shuffle=False, log_every_n_batches=100):        
    # Define Loss Function Use (Cross-Entropy Loss)
    test_accuracies = []
    train_accuracies = []
    loss_fn = nn.CrossEntropyLoss()
    # Set up optimization
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    # Load dataset
    train_set, test_set = load_dataset(batch_size, augmentation=augmentation, shuffle=shuffle)
    for e in range(epochs):
        print(f"Epoch {e+1}/{epochs}\n-------------------------------")
        train_loop(train_set, model, loss_fn, optimizer, log_every_n_batches=log_every_n_batches)
        train_acc, train_loss = accuracy(train_set, model, loss_fn)
        test_acc, test_loss = accuracy(test_set, model, loss_fn)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        lr_scheduler.step()
    return train_accuracies, test_accuracies

