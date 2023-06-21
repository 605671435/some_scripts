import torch.nn as nn
class Encoder_Decoder(nn.Module):
    def __init__(self,
                 backbone,
                 decode_head):
        super(Encoder_Decoder, self).__init__()
        self.backbone = backbone
        self.decode_head = decode_head
    def forward(self, x, image=None):
        x = self.backbone.forward(x, image)
        out = self.decode_head.forward(x)
        return out