import torch
import torch.nn as nn
from torch import Tensor

class HardDistillationLoss(nn.Module):
    def __init__(self, teacher: nn.Module):
        super().__init__()
        self.teacher = teacher
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, inputs: Tensor, outputs : Tensor, labels: Tensor) -> Tensor:
        base_loss = self.criterion(outputs, labels)

        with torch.no_grad():
            teacher_outputs = self.teacher(inputs)
        teacher_labels = torch.argmax(teacher_outputs, dim=1)
        teacher_loss = self.criterion(outputs, teacher_labels)
        
        return 0.5 * base_loss + 0.5 * teacher_loss