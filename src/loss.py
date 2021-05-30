import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, normalize_rows=False):
        super(Loss, self).__init__()
        self.normalize_rows = normalize_rows

    def forward(self, all_out1, all_out2):
        loss = 0.0
        for idx, (out1, out2) in enumerate(zip(all_out1, all_out2)):
            softmax_view1_col = F.softmax(out1.clone(), dim=0)
            softmax_view2_col = F.softmax(out2.clone(), dim=0)

            if self.normalize_rows:
                softmax_view1_col = F.normalize(softmax_view1_col, p=1, dim=1)
                softmax_view2_col = F.normalize(softmax_view2_col, p=1, dim=1)

            log_softmax_view1_row = F.log_softmax(out1, dim=1)
            log_softmax_view2_row = F.log_softmax(out2, dim=1)

            # cross entropy
            loss_view1 = -1.0 * (softmax_view1_col * log_softmax_view2_row).sum() / softmax_view1_col.shape[0]
            loss_view2 = -1.0 * (softmax_view2_col * log_softmax_view1_row).sum() / softmax_view2_col.shape[0]
            loss_i = 0.5 * (loss_view1 + loss_view2)

            loss += loss_i

            # # debug
            # if idx == 0:
            #     target = all_out2[idx].argmax(dim=1)
            #     print('number of unique assignments (cls {}): {}'.format(idx, th.unique(target).shape[0]))
            #     print('loss_0', loss_i)

        loss /= len(all_out1)

        return loss
