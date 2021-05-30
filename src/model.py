import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, base_model, dim=128, hidden_dim=2048, cls_size=3000, tau=0.07, num_cls=10,
                 no_mlp=False, num_hidden=1, learnable_cls=False, backbone_dim=2048):
        super(Model, self).__init__()
        self.cls_size = cls_size
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.num_cls = num_cls
        self.no_mlp = no_mlp
        self.num_hidden = num_hidden
        self.learnable_cls = learnable_cls
        self.backbone_dim = backbone_dim

        # backbone
        self.backbone = base_model()

        if self.no_mlp:
            self.backbone.fc = nn.Identity()
        else:
            # projection
            if self.num_hidden == 1:
                self.backbone.fc = nn.Sequential(
                    nn.Linear(self.backbone_dim, self.hidden_dim),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(self.hidden_dim, self.dim)
                )
            elif self.num_hidden == 2:
                self.backbone.fc = nn.Sequential(
                    nn.Linear(self.backbone_dim, self.hidden_dim),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(self.hidden_dim, self.dim)
                )
            elif self.num_hidden == 3:
                self.backbone.fc = nn.Sequential(
                    nn.Linear(self.backbone_dim, self.hidden_dim),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(self.hidden_dim, self.dim)
                )
            else:
                raise Exception('Number of MLP hidden layers must be 1, 2 or 3!')

        # classifiers
        for cls in range(self.num_cls):
            if self.cls_size > self.dim:
                cls_i, _ = np.linalg.qr(np.random.randn(self.cls_size, self.dim))
            else:
                cls_i, _ = np.linalg.qr(np.random.randn(self.dim, self.cls_size))
                cls_i = cls_i.T
            setattr(self, "cls_%d" % cls,
                    nn.Parameter(th.from_numpy(np.float32(cls_i)), requires_grad=self.learnable_cls))

    def forward(self, view1, view2=None):
        if view2 is not None:
            view1 = F.normalize(self.backbone(view1))
            view2 = F.normalize(self.backbone(view2))

            cls_view1 = []
            cls_view2 = []

            for cls in range(self.num_cls):
                cls_view1.append(F.linear(view1, F.normalize(getattr(self, "cls_%d" % cls))) / self.tau)
                cls_view2.append(F.linear(view2, F.normalize(getattr(self, "cls_%d" % cls))) / self.tau)
        else:
            view1 = F.normalize(self.backbone(view1))
            cls_view1 = F.linear(view1, F.normalize(getattr(self, "cls_0"))) / self.tau
            return cls_view1

        return cls_view1, cls_view2
