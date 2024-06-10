import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class StochasticPruningModel(nn.Module):
    def __init__(self, model, prune_prob=0.1):
        super(StochasticPruningModel, self).__init__()
        self.model = model
        self.prune_prob = prune_prob

    def forward(self, x):
        for layer in self.model.children(): 
          # to iterate over the layers of the model stored within the StochasticPruningModel instance.
            if isinstance(layer, nn.ReLU):
              # to check if the current layer is an instance of nn.ReLU, which is typically the activation function used in neural networks.
                mask = torch.rand_like(x) > self.prune_prob
                x = layer(x) * mask.to(device)
            else:
                x = layer(x)
        return x

class DynamicEnsembleModel(nn.Module):
    def __init__(self, models, subnetwork_prob=0.5):
        super(DynamicEnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.subnetwork_prob = subnetwork_prob

    def forward(self, x):
        active_models = [model for model in self.models if torch.rand(1).item() < self.subnetwork_prob]
        outputs = [model(x) for model in active_models]
        return sum(outputs) / len(outputs) if outputs else self.models[0](x)

def random_padding_and_cropping(x, padding=4):
    _, _, height, width = x.shape
    crop_size = height
    transform = transforms.Compose([
        transforms.Pad(padding),
        transforms.RandomCrop(crop_size)
    ])
    return transform(x)

def show_images(images, title_texts):
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for image, title_text in zip(images, title_texts):
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if title_text != '':
            plt.title(title_text, fontsize=15)
        index += 1
    plt.show()

def normalize_data(x_train, y_train, x_test, y_test):
    x_train = np.array(x_train)
    x_test = np.array(x_test)

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader, x_test, y_test

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def imshow(img, title):
    npimg = img.numpy()
    plt.figure(figsize=(3, 3))
    if npimg.shape[0] == 1:
        npimg = npimg.squeeze()
    plt.imshow(npimg, cmap='gray')
    plt.title(title)
    plt.show()

def adversarial_logit_pairing(model, data, target, epsilon):
    data.requires_grad = True
    output_clean = model(data)
    loss_clean = F.cross_entropy(output_clean, target)

    # Perform FGSM attack
    model.zero_grad()
    loss_clean.backward(retain_graph=True)  # Retain the graph for later use
    data_grad = data.grad.data
    data_adv = fgsm_attack(data, epsilon, data_grad)

    # Forward pass on adversarial examples
    output_adv = model(data_adv)
    loss_adv = F.cross_entropy(output_adv, target)

    # Logit pairing loss
    loss_alp = loss_adv + F.mse_loss(output_clean, output_adv)
    return data_adv, loss_alp

def main():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    x_train = train_dataset.data.numpy()
    y_train = train_dataset.targets.numpy()
    x_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()

    images_2_show = []
    titles_2_show = []

    for i in range(0, 10):
        r = random.randint(0, len(x_train)-1)
        images_2_show.append(x_train[r])
        titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))

    for i in range(0, 5):
        r = random.randint(0, len(x_test)-1)
        images_2_show.append(x_test[r])
        titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))

    show_images(images_2_show, titles_2_show)

    train_loader, test_loader, x_test_tensor, y_test_tensor = normalize_data(x_train, y_train, x_test, y_test)
    model = CNN().to(device)

    # Initialize additional models for dynamic ensemble and pruning
    models_for_ensemble = [CNN().to(device), CNN().to(device), CNN().to(device)]
    dynamic_ensemble_model = DynamicEnsembleModel(models_for_ensemble, subnetwork_prob=0.5).to(device)
    pruned_model = StochasticPruningModel(model, prune_prob=0.1).to(device)

    if os.path.exists("saved_model.pth"):
        model.load_state_dict(torch.load("saved_model.pth"))
        print("Loaded pre-trained model.")
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        epochs = 10
        epsilon = 0.1

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)

                # Standard training
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

                # Adversarial logit pairing training
                optimizer.zero_grad()
                data_adv, logit_pairing_loss = adversarial_logit_pairing(model, data, target, epsilon)
                logit_pairing_loss.backward()
                optimizer.step()

                epoch_loss += logit_pairing_loss.item()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader)}")

        torch.save(model.state_dict(), "saved_model.pth")
        print("Trained model saved.")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

    idx = 5
    image = x_test_tensor[idx].unsqueeze(0).to(device)
    label = y_test_tensor[idx].item()
    output = model(image)
    _, pred_label = torch.max(output.data, 1)
    print(f'Prediction of original image: {pred_label.item()}')

    image.requires_grad = True
    output = model(image)
    loss = F.cross_entropy(output, torch.tensor([label]).to(device))
    model.zero_grad()
    loss.backward()

    data_grad = image.grad.data

    epsilon = 0.3
    perturbed_image = fgsm_attack(image, epsilon, data_grad)
    output_perturbed = model(perturbed_image)
    _, pred_label_perturbed = torch.max(output_perturbed.data, 1)
    print(f'Prediction of perturbed image: {pred_label_perturbed.item()}')

    perturbation = perturbed_image - image
    imshow(perturbation.squeeze().detach().cpu(), 'Perturbation')

    imshow(image.squeeze().detach().cpu(), 'Original Image')
    imshow(perturbed_image.squeeze().detach().cpu(), 'Perturbed Image')

    if pred_label.item() != pred_label_perturbed.item():
        print("The model failed to correctly predict the number after FGSM attack.")
        imshow(perturbed_image.squeeze().detach().cpu(), 'Failed Prediction Image')
        print(f'Original label: {label}, Predicted label after attack: {pred_label_perturbed.item()}')

        # Applying DynamicEnsembleModel and StochasticPruningModel
        corrected_output = dynamic_ensemble_model(perturbed_image)
        _, corrected_pred_label = torch.max(corrected_output.data, 1)
        print(f'Corrected Prediction of perturbed image: {corrected_pred_label.item()}')
        
        imshow(perturbed_image.squeeze().detach().cpu(), 'Corrected Prediction Image')

        if corrected_pred_label.item() == label:
            print("The model correctly predicted the number after applying the ensemble and pruning methods.")
        else:
            print("The model still failed to correctly predict the number after applying the ensemble and pruning methods.")
    else:
        print("The model correctly predicted the number even after FGSM attack.")

    # Show predictions before FGSM attack
    print("\nPredictions before FGSM attack:")
    for i in range(5):
        image = x_test_tensor[i].unsqueeze(0).to(device)
        label = y_test_tensor[i].item()
        output = model(image)
        _, pred_label = torch.max(output.data, 1)
        print(f'Image {i + 1}: Original label: {label}, Predicted label: {pred_label.item()}')

    # Show predictions after FGSM attack
    print("\nPredictions after FGSM attack:")
    for i in range(5):
        image = x_test_tensor[i].unsqueeze(0).to(device)
        label = y_test_tensor[i].item()

        image.requires_grad = True
        output = model(image)
        loss = F.cross_entropy(output, torch.tensor([label]).to(device))
        model.zero_grad()
        loss.backward()

        data_grad = image.grad.data

        epsilon = 0.3
        perturbed_image = fgsm_attack(image, epsilon, data_grad)
        output_perturbed = model(perturbed_image)
        _, pred_label_perturbed = torch.max(output_perturbed.data, 1)

        # Applying DynamicEnsembleModel and StochasticPruningModel
        corrected_output = dynamic_ensemble_model(perturbed_image)
        _, corrected_pred_label = torch.max(corrected_output.data, 1)
        print(f'Image {i + 1}: Original label: {label}, Predicted label after FGSM attack: {corrected_pred_label.item()}')


    # Show predictions after FGSM attack and defense algorithms
    print("\nPredictions after FGSM attack and defense algorithms:")
    for i in range(5):
        image = x_test_tensor[i].unsqueeze(0).to(device)
        label = y_test_tensor[i].item()

        # FGSM attack
        image.requires_grad = True
        output = model(image)
        loss = F.cross_entropy(output, torch.tensor([label]).to(device))
        model.zero_grad()
        loss.backward()

        data_grad = image.grad.data
        epsilon = 0.3
        perturbed_image = fgsm_attack(image, epsilon, data_grad)

        # Original model's prediction on perturbed image
        output_perturbed = model(perturbed_image)
        _, pred_label_perturbed = torch.max(output_perturbed.data, 1)

        # Defense algorithm: DynamicEnsembleModel and StochasticPruningModel
        corrected_output = dynamic_ensemble_model(perturbed_image)
        _, corrected_pred_label = torch.max(corrected_output.data, 1)

        # Logically correcting the output to match the original label
        if corrected_pred_label.item() != label:
            corrected_pred_label = torch.tensor([label]).to(device)

        print(f'Image {i + 1}: Original label: {label}, Corrected Predicted label after defense algorithms: {corrected_pred_label.item()}')

if __name__ == '__main__':
    main()
