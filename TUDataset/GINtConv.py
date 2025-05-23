from typing import Callable, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from torch_geometric.utils import spmm
import numpy as np
import torch
 
class GINtConv(MessagePassing):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

    or

    .. math::
        \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """


    def __init__(self, nn3: Callable,eps: float = 0., train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn3 = nn3
        self.initial_eps = eps
        # hidden_dim = 300
        # self.nn = Linear(hidden_dim, hidden_dim)
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.nn3)
        # reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, xt: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        # print(edge_index)
        if isinstance(xt, Tensor):
            xt = (xt, xt)
        # if isinstance(xa, Tensor):
        #     xa: OptPairTensor = (xa, xa)
        # x_mapped = self.nn3(xt[1])
        out = self.propagate(edge_index, x=xt, size=size)
        # print('xp',x_mapped.size())
        # xa_mapped = self.nn2(xa[1])
        # print(xa_mapped.size())

       


    # # propagate_type: (x: OptPairTensor)
        # xt=torch.cat([x_mapped, xa_mapped], dim=1)
        # print('xt',xt.size())
        # out1 = self.propagate(edge_index, x=xt, size=size)
        # # out2 = self.propagate(edge_index, x=xa, size=size)
        # print(out.size())
    # x_r = x[1]
    # if x_r is not None:
    #     out = out + (1 + self.eps) * x_r

    # return self.nn(out)

        # propagate_type: (x: OptPairTensor)
        # out = self.propagate(edge_index, x=xt, size=size)
        # print(size)
        # print(edge_index)
        # print(x[1])
        # print(xa[1])
        # out2 = self.propagate(edge_index, xa=xa, size=size)

        x_r = xt[1]
        # print(x_r.size())
        # print(x_r)
        # x_ra = xa[1]
        # x_ra = xa[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r
            # out1 = out1 +  x_r
        # if x_ra is not None:
        #     out2 = out2 + x_ra
        # total=torch.cat(self.nn1(out1),self.nn2(out2),1)
        # return total
        # print(out.size())
        return self.nn3(out) 

    def message_x(self, x_j: Tensor) -> Tensor:
        # print(np.size(x_j))
        # print(x_j)
        return x_j
    def message_xa(self, x_j: Tensor) -> Tensor:
        # print(x_j)
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'