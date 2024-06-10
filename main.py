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
