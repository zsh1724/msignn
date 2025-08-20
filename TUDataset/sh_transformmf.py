import torch

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
import copy
from torch import Tensor

from torch_geometric.data.datapipes import functional_transform
from get_msf import get_msf

 
@functional_transform('msf')
class msf(BaseTransform):
    def forward(self, data: Data) -> Data:
        i=3 #k in k-nmi

        """
        get_metric_space() -> add_nodes, add_edges
        """
        msf = get_msf(data.edge_index,data.num_nodes,i)
        data.xa = msf
        xab = torch.cat((data.x,msf),1)

        return data
    def __call__(self, data: Data) -> Data:
        return self.forward(data)
    


