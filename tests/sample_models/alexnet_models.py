import torch

__all__ = ["AlexNet", "alexnet"]


class AlexNet(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = torch.nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = torch.nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        # This is simplified (original AlexNet contains one more linear 4096-4096)
        self.l1 = torch.nn.Linear(6400, 4096)
        self.d1 = torch.nn.Dropout(p=0.5)
        self.l2 = torch.nn.Linear(4096, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)

        # x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = torch.nn.functional.relu(x)

        # x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = torch.nn.functional.relu(x)

        x = self.conv4(x)
        x = torch.nn.functional.relu(x)

        x = self.conv5(x)
        x = torch.nn.functional.relu(x)

        # x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        x = self.maxpool3(x)

        x = torch.flatten(x, start_dim=1)
        x = self.l1(x)
        x = torch.nn.functional.relu(x)
        x = self.d1(x)
        x = self.l2(x)
        return x


def alexnet(num_classes: int) -> torch.nn.Module:
    return AlexNet(num_classes=num_classes)
