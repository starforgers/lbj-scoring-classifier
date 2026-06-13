import torch
import torch.nn as nn
import torch.nn.init


class NeuralNetwork(nn.Module):

    def __init__(
        self,
        input_size,
        num_classes,
        list_hidden=None,
        activation='sigmoid',
        dropout_p=0.2,
    ):
        super(NeuralNetwork, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.list_hidden = list_hidden if list_hidden is not None else [32, 16]
        self.activation = activation
        self.dropout_p = dropout_p
        self.layers = None

    def create_network(self):
        layers = []

        in_features = self.input_size
        for hidden_units in self.list_hidden:
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(nn.BatchNorm1d(hidden_units))
            layers.append(self.get_activation(self.activation))
            layers.append(nn.Dropout(p=self.dropout_p))
            in_features = hidden_units

        layers.append(nn.Linear(in_features, 1))

        self.layers = nn.Sequential(*layers)

    def init_weights(self, seed=2):
        torch.manual_seed(seed)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def get_activation(self, mode='sigmoid'):
        activation = nn.Sigmoid()

        if mode == 'tanh':
            activation = nn.Tanh()
        elif mode == 'relu':
            activation = nn.ReLU(inplace=True)

        return activation

    def forward(self, x, verbose=False):
        if self.layers is None:
            raise ValueError('Call create_network() before forward().')

        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if verbose:
                print('Output of layer ' + str(i))
                print(x, '\n')

        logits = x
        probabilities = torch.sigmoid(logits)

        if verbose:
            print('Sigmoid(logits)')
            print(probabilities, '\n')

        return logits, probabilities

    def predict(self, probabilities):
        return (probabilities >= 0.5).long().view(-1)
