class DynamicEnsembleModel(nn.Module):
    def __init__(self, models, subnetwork_prob=0.5):
        super(DynamicEnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.subnetwork_prob = subnetwork_prob

    def forward(self, x):
        active_models = [model for model in self.models if torch.rand(1).item() < self.subnetwork_prob]
        outputs = [model(x) for model in active_models]
        return sum(outputs) / len(outputs) if outputs else self.models[0](x)
