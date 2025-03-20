import os
import torch
from models import (
    Autoformer,
    Transformer,
    DLinear,
    GraphPatchTST,
    GRU,
    CNN1D,
    wo_global_embedding,
    wo_graph_learning,
)


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "Autoformer": Autoformer,
            "Transformer": Transformer,
            "DLinear": DLinear,
            "GraphPatchTST": GraphPatchTST,
            "GRU": GRU,
            "CNN1D": CNN1D,
            "wo_global_embedding": wo_global_embedding,
            "wo_graph_learning": wo_graph_learning,
        }
        if args.model == "Mamba":
            print("Please make sure you have successfully installed mamba_ssm")
            from models import Mamba

            self.model_dict[Mamba] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
