import torch as t
import torch.nn.functional as F


class Net(t.nn.Module):
    def __init__(self, dropouts: tuple[float, float]):
        super().__init__()
        self.conv1 = t.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = t.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = t.nn.Dropout(dropouts[0])
        self.dropout2 = t.nn.Dropout(dropouts[1])
        self.fc1 = t.nn.Linear(9216, 128)
        self.fc2 = t.nn.Linear(128, 10)

    # x \in -1 x 1 x 28 x 28
    # output \in -1 x 10
    def forward(self, x) -> t.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(F.max_pool2d(x, 2))
        x = t.flatten(x, 1)
        x = self.dropout2(F.relu(self.fc1(x)))
        x = self.fc2(x)
        outputs = F.log_softmax(x, dim=1)
        return outputs
