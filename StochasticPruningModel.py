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
