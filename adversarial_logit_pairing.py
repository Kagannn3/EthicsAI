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
