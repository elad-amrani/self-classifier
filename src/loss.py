import torch as th
import torch.nn as nn
import torch.nn.functional as F
import utils


class Loss(nn.Module):
    def __init__(self, row_tau=0.1, col_tau=0.1, eps=1e-8):
        super(Loss, self).__init__()
        self.row_tau = row_tau
        self.col_tau = col_tau
        self.eps = eps

    def forward(self, out):
        total_loss = 0.0
        num_loss_terms = 0

        for cls_idx, cls_out in enumerate(out):  # classifiers
            # gather samples from all workers
            cls_out = [utils.AllGather.apply(x).float() for x in cls_out]

            const = cls_out[0].shape[0] / cls_out[0].shape[1]
            target = []

            for view_i_idx, view_i in enumerate(cls_out):
                view_i_target = F.softmax(view_i / self.col_tau, dim=0)
                view_i_target = utils.keep_current(view_i_target)
                view_i_target = F.normalize(view_i_target, p=1, dim=1, eps=self.eps)
                target.append(view_i_target)

            for view_j_idx, view_j in enumerate(cls_out):  # view j
                view_j_pred = F.softmax(view_j / self.row_tau, dim=1)
                view_j_pred = F.normalize(view_j_pred, p=1, dim=0, eps=self.eps)
                view_j_pred = utils.keep_current(view_j_pred)
                view_j_log_pred = th.log(const * view_j_pred + self.eps)

                for view_i_idx, view_i_target in enumerate(target):

                    if view_i_idx == view_j_idx or (view_i_idx >= 2 and view_j_idx >= 2):
                        # skip cases when it's the same view, or when both views are 'local' (small)
                        continue

                    # cross entropy
                    loss_i_j = - th.mean(th.sum(view_i_target * view_j_log_pred, dim=1))
                    total_loss += loss_i_j
                    num_loss_terms += 1

        total_loss /= num_loss_terms

        return total_loss
