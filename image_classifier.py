import torch.nn as nn
class ImageClassifier(nn.Module):
    def __init__(self,
                 backbone,
                 neck,
                 head):
        super(ImageClassifier, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x, image):
        x = self.backbone.forward(x, image)
        if self.neck is not None:
            x = self.neck.forward(x[3])
        out = self.head.forward(x)
        return out