import torch
import torch.nn as nn
from tqdm import tqdm
from aggregation.aggregator import Response_Aggregator

class AggregatorTrainer(object):
    def __init__(self, num_class, lrate=1e-3, lrate_decay=250e3, ckpt_path=None, input_class=None):
        super(AggregatorTrainer, self).__init__()
        self.num_class = num_class
        self.lrate = lrate
        self.lrate_decay = lrate_decay
        self.ckpt_path = ckpt_path
        self.input_class = input_class

    def create_model(self):
        model = Response_Aggregator(num_class=self.num_class, W=128, input_class=self.input_class)
        grad_vars = list(model.parameters())
        # Create optimizer
        optimizer = torch.optim.Adam(params=grad_vars, lr=self.lrate)
        start = 0
        self.agg_net = model
        self.optimizer = optimizer
        return start

    def step(self, global_step, response_logits, gt_labels):
        '''
        response_logits: [N, num_class]
        gt_labels: [N]
        '''
        CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=-1)
        crossentropy_loss = lambda logit, label: CrossEntropyLoss(logit, label)
        self.optimizer.zero_grad()
        segmentation_logits = self.agg_net(response_logits)
        aggregation_loss = crossentropy_loss(segmentation_logits, gt_labels)
        aggregation_loss.backward()
        self.optimizer.step()

        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = self.lrate_decay
        new_lrate = self.lrate * (decay_rate ** (global_step / decay_steps))
        print('lrate:', new_lrate)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lrate

        tqdm.write(f"[Training Metric] Iter: {global_step} "
                   f"aggregation_loss: {aggregation_loss.item()}")
        return aggregation_loss.item()