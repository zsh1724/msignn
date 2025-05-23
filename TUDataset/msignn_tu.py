import argparse
import os.path as osp
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import MLP, global_add_pool
from sh_transformmf import msf
from GINtConv import GINtConv
from GINeConv import GINConv

'''
MUTAG PROTEIN PTC_MR NCI1 NCI109 (max number of nodes)
28 620 64 111 111
''' 

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MUTAG')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cpu')

init_wandb(name=f'GIN-{args.dataset}', batch_size=args.batch_size, lr=args.lr,
           epochs=args.epochs, hidden_channels=args.hidden_channels,
           num_layers=args.num_layers, device=device)

path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset', 'TU')




class Net(torch.nn.Module):
    def __init__(self, in_channels,num_featuresa, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            mlp = MLP([in_channels, hidden_channels//2, hidden_channels//2])
            mlp2 = MLP([num_featuresa, hidden_channels//2, hidden_channels//2])
            mlp3 = MLP([hidden_channels, hidden_channels, hidden_channels])
            if i==0:
                self.convs.append(GINConv(nn1=mlp,nn2=mlp2, train_eps=False))
            else:
                self.convs.append(GINtConv(nn3=mlp3, train_eps=False))

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.1)

    def forward(self, x,xa, edge_index, batch):
        for i,conv in enumerate(self.convs):
            if i== 0:
                xt = conv(x, xa, edge_index).relu()
            else :
                xt = conv(xt, edge_index).relu()
        xt = global_add_pool(xt, batch)
        return self.mlp(xt)

def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.xa, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.xa, data.edge_index, data.batch).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)

dataset = TUDataset(path, name=args.dataset, pre_transform = msf()).shuffle()

kf = KFold(n_splits=10, shuffle=True)

# if args.dataset=='MUTAG':
#     n_fa = 29
# if args.dataset=='PROTEIN':   
#     n_fa = 621
# if args.dataset=='PTC_MR':
#     n_fa = 29
# if args.dataset=='NCI1':   
#     n_fa = 621
# if args.dataset=='NCI1':   
#     n_fa = 621

num_featuresa=29

accs=[]
for train_index, test_index in kf.split(dataset):
    model = Net(dataset.num_features, num_featuresa, args.hidden_channels, dataset.num_classes,
                args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.5, last_epoch=-1)
    for epoch in range(1, args.epochs + 1):
        


        train_dataset=[dataset[i] for i in train_index]
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
        test_dataset=[dataset[i] for i in test_index]
        test_loader = DataLoader(test_dataset, args.batch_size)
        loss = train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        log(Epoch=epoch, Loss=loss, Train=train_acc, Test=test_acc)

        