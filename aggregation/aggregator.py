import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn

def fc_block(in_f, out_f):
    return torch.nn.Sequential(
        torch.nn.Linear(in_f, out_f),
        torch.nn.ReLU(out_f)
    )

class Response_Aggregator(nn.Module):
    # maps from response logits [num_class] to segmentation logits [num_class]
    def __init__(self, num_class, W=256, input_class=None):
        super(Response_Aggregator, self).__init__()
        self.num_class = num_class
        self.W = W

        # self.input_linear = fc_block(num_class, W)
        if input_class is not None:
            self.input_linear = nn.Sequential(fc_block(input_class, W), fc_block(W, W // 2))
        else:
            self.input_linear = nn.Sequential(fc_block(num_class, W), fc_block(W, W // 2))
        self.output_linear = nn.Linear(W // 2, num_class)
        # self.output_linear = nn.Linear(W, num_class)

    def forward(self, x):
        h = self.input_linear(x)
        seg_logits = self.output_linear(h)
        return seg_logits
