'''
****************NOTE*****************
CREDITS : Thomas Kipf
since datasets are the same as those in kipf's implementation, 
Their preprocessing source was used as-is.
*************************************
'''
import sys
import torch
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch.nn.functional as F
from torch_geometric.datasets import WikipediaNetwork, WebKB, Amazon, AttributedGraphDataset, WikiCS, PPI, CitationFull, IMDB, Twitch, LastFMAsia
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_geometric.utils import to_undirected, from_scipy_sparse_matrix,dense_to_sparse,is_undirected
from torch_geometric.nn import Node2Vec
import torch
from ogb.linkproppred import PygLinkPropPredDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def CalN2V(edge_index, dim, in_out_param):
    modeljure = Node2Vec(edge_index, embedding_dim=dim, walk_length=20,
                     context_size=10, walks_per_node=1,
                     num_negative_samples=1, p=1, q=in_out_param, sparse=True).to(device)
    loader = modeljure.loader(batch_size=32, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(modeljure.parameters()), lr=0.01)
    modeljure.train()
    total_loss = 0
    print('___Calculating Node2Vec features___')
    for i in range(201):
        total_loss=0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = modeljure.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if i%20 == 0:
            print(f'Setp: {i:03d} /200, Loss : {total_loss:.4f}')
    output=(modeljure.forward()).cpu().clone().detach()
    del modeljure
    del loader
    torch.cuda.empty_cache()
    return output

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset):
    if dataset in ['cora', 'citeseer', 'pubmed']:
        # load the data: x, tx, allx, graph
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("/home/retro/SECRET/data/citation/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        x, y, tx, ty, allx, ally, graph = tuple(objects)
        
        test_idx_reorder = parse_index_file("/home/retro/SECRET/data/citation/ind.{}.test.index".format(dataset))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        # labels = np.argmax(labels, axis=1)

        idx_train = list(range(len(y)))
        idx_val = list(range(len(y), len(y)+500))
        idx_test = test_idx_range.tolist()
    elif dataset in ['ogbl-ddi','ogbl-collab','ogbl-ppa','ogbl-citation2']:
        """ogbdataset = PygLinkPropPredDataset(name = dataset) 
        # split_edge = dataset.get_edge_split()
        # train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
        split_idx = ogbdataset.get_edge_split()
        graph = ogbdataset[0]
        if(dataset=='ogbl-ddi'):
            # features = sp.csr_matrix(torch.ones((graph['num_nodes'], 1)))
            features = sp.lil_matrix(torch.ones((graph['num_nodes'], 1)))
        else:    
            # features = sp.csr_matrix(graph['x'])
            features = sp.lil_matrix(graph['x'].numpy())
        print(f"Number node:{graph['num_nodes']}")
        # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        row = graph['edge_index'][0]
        col = graph['edge_index'][1]
        # data = np.ones(len(row))  # 假設每個邊的權重都是 1
        # adj = to_scipy_sparse_matrix(split_idx['train']['edge'].t(), num_nodes=graph.num_nodes)
        adj = to_scipy_sparse_matrix(graph.edge_index, num_nodes=graph.num_nodes)
        # adj = sp.coo_matrix((data, (row, col)), shape=(graph['num_nodes'], graph['num_nodes']))
        # adj = sp.coo_matrix((graph['edge_index'][1], graph['edge_index'][0]), shape=(graph['num_nodes'], graph['num_nodes']))
        labels = None #graph['node_label']

        # 重新排列索引
        idx_train = split_idx['train']
        idx_val = split_idx['valid']
        idx_test = split_idx['test']
        """
        print(f"Loading dataset: {dataset}")

        ogbdataset = PygLinkPropPredDataset(name=dataset)
        split_idx = ogbdataset.get_edge_split()
        graph = ogbdataset[0]

        # Display keys and number of nodes
        print("Keys in graph:", graph.keys)
        num_nodes = graph['num_nodes']
        print(f"Number of nodes (original): {num_nodes}")

        # Node subsampling: Randomly select a subset of nodes
        subsample_ratio = 0.0001  # Adjust this ratio as needed
        num_subsampled_nodes = int(num_nodes * subsample_ratio)
        sampled_node_indices = np.random.choice(num_nodes, num_subsampled_nodes, replace=False)

        # Create a mapping from original indices to subsampled indices
        index_mapping = {original: new for new, original in enumerate(sampled_node_indices)}
        print(f"Sampled node indices: {sampled_node_indices}")

        # Default features since no features are available in the dataset
        if dataset == 'ogbl-ddi':
            features = sp.lil_matrix(torch.ones((num_nodes, 1)))  # Adjust if features are required
        else:
            features = sp.lil_matrix(graph['x'].numpy())

        # Subsample the features matrix to include only the sampled nodes
        features = features[sampled_node_indices]  # Keep only the rows corresponding to sampled nodes
        print(f"Features shape after subsampling: {features.shape}")

        # Create new edge index based on sampled nodes
        edge_index = graph['edge_index'].numpy()
        filtered_edge_index = []

        # Filter edges to only include sampled nodes
        for edge in edge_index.T:
            if edge[0] in index_mapping and edge[1] in index_mapping:
                filtered_edge_index.append((index_mapping[edge[0]], index_mapping[edge[1]]))

        filtered_edge_index = np.array(filtered_edge_index).T

        # Convert filtered edge index to a PyTorch tensor
        filtered_edge_index = torch.tensor(filtered_edge_index, dtype=torch.long)
        print(f"Filtered edge index shape: {filtered_edge_index.shape}")

        # Update adjacency matrix using the filtered edge index
        adj = to_scipy_sparse_matrix(filtered_edge_index, num_nodes=num_subsampled_nodes)
        print(f"Adjacency matrix shape after filtering: {adj.shape}")

        # Print split indices to debug
        #print("Split indices:", split_idx)

        # Adjust index mappings for train, val, test
        idx_train = {'edge': [index_mapping[i.item()] for i in split_idx['train']['edge'].flatten() if i.item() in index_mapping]}
        idx_val = {'edge': [index_mapping[i.item()] for i in split_idx['valid']['edge'].flatten() if i.item() in index_mapping]}
        idx_test = {'edge': [index_mapping[i.item()] for i in split_idx['test']['edge'].flatten() if i.item() in index_mapping]}

        # Convert lists to tensors
        idx_train['edge'] = torch.tensor(idx_train['edge'], dtype=torch.long)
        idx_val['edge'] = torch.tensor(idx_val['edge'], dtype=torch.long)
        idx_test['edge'] = torch.tensor(idx_test['edge'], dtype=torch.long)
        
        labels = None #graph['node_label']

        print(f"Number of nodes (subsampled): {num_subsampled_nodes}")
        print(f"Training indices: {len(idx_train['edge'])}")
        print(f"Validation indices: {len(idx_val['edge'])}")
        print(f"Test indices: {len(idx_test['edge'])}")

    elif dataset in ['chameleon', 'crocodile', 'squirrel']:
        if dataset in ['crocodile', 'squirrel']:
            data = WikipediaNetwork('data/', dataset)[0]
        elif dataset == 'chameleon':
            tmp_feature = WikipediaNetwork('data/', dataset, geom_gcn_preprocess = False)[0].x
            data = WikipediaNetwork('data/', dataset)[0]
            data.x = tmp_feature

        adj = to_scipy_sparse_matrix(data.edge_index)
        features = sp.lil_matrix(data.x.numpy())
        labels = F.one_hot(data.y).numpy()
        idx_train = torch.arange(data.x.shape[0])[data.train_mask[:, 0]].tolist()
        idx_val = torch.arange(data.x.shape[0])[data.val_mask[:, 0]].tolist()
        idx_test = torch.arange(data.x.shape[0])[data.test_mask[:, 0]].tolist()

    elif dataset in ['cornell', 'texas', 'wisconsin']:
        data = WebKB('data/', dataset.capitalize())[0]
        adj = to_scipy_sparse_matrix(data.edge_index)
        features = sp.lil_matrix(data.x.numpy())

        labels = F.one_hot(data.y).numpy()
        idx_train = torch.arange(data.x.shape[0])[data.train_mask[:, 0]].tolist()
        idx_val = torch.arange(data.x.shape[0])[data.val_mask[:, 0]].tolist()
        idx_test = torch.arange(data.x.shape[0])[data.test_mask[:, 0]].tolist()

    elif dataset in ['amazon_photo']:
        data = Amazon('data/amazon_photo', 'Photo')[0]
        adj = to_scipy_sparse_matrix(data.edge_index)
        features = sp.lil_matrix(data.x.numpy())
        labels = F.one_hot(data.y).numpy()

        idx_train = []
        idx_val = []
        for i in range(int(max(data.y))+1):
            target = (data.y == i).nonzero(as_tuple=True)[0]
            indices = torch.randperm(len(target))[:20]
            idx_train.extend(target[indices].numpy().tolist())
        
        unselected = torch.ones(data.x.shape[0], dtype=torch.bool)
        unselected[torch.tensor(idx_train)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        indices = torch.randperm(len(target))[:500]
        idx_val.extend(target[indices].numpy().tolist())

        unselected[torch.tensor(idx_val)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        idx_test = target.numpy().tolist()

    elif dataset in ['Facebook']:
        data = AttributedGraphDataset('data/Facebook', dataset)[0]
        adj = to_scipy_sparse_matrix(data.edge_index)
        features = sp.lil_matrix(data.x.numpy())
        labels = (data.y).numpy()
        idx_train = []
        idx_val = []
        for i in range(data.y.shape[1]):
            target = (data.y == i).nonzero(as_tuple=True)[0]
            indices = torch.randperm(len(target))[:20]
            idx_train.extend(target[indices].numpy().tolist())
        
        unselected = torch.ones(data.x.shape[0], dtype=torch.bool)
        unselected[torch.tensor(idx_train)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        indices = torch.randperm(len(target))[:500]
        idx_val.extend(target[indices].numpy().tolist())

        unselected[torch.tensor(idx_val)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        idx_test = target.numpy().tolist()

    elif dataset in ['Flickr']:
        data = AttributedGraphDataset('data/Flickr', dataset)[0]
        data.x = data.x.to_dense()
        adj = to_scipy_sparse_matrix(data.edge_index)
        features = sp.lil_matrix(data.x.numpy())
        labels = F.one_hot(data.y).numpy()
        idx_train = []
        idx_val = []
        for i in range(int(max(data.y))+1):
            target = (data.y == i).nonzero(as_tuple=True)[0]
            indices = torch.randperm(len(target))[:20]
            idx_train.extend(target[indices].numpy().tolist())
        
        unselected = torch.ones(data.x.shape[0], dtype=torch.bool)
        unselected[torch.tensor(idx_train)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        indices = torch.randperm(len(target))[:500]
        idx_val.extend(target[indices].numpy().tolist())

        unselected[torch.tensor(idx_val)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        idx_test = target.numpy().tolist()

    elif dataset in ['WikiCS']:
        data = WikiCS('data/WikiCS')[0]
        adj = to_scipy_sparse_matrix(data.edge_index)
        features = sp.lil_matrix(data.x.numpy())
        labels = F.one_hot(data.y).numpy()
        idx_train = torch.arange(data.x.shape[0])[data.train_mask[:, 0]].tolist()
        idx_val = torch.arange(data.x.shape[0])[data.val_mask[:, 0]].tolist()
        idx_test = torch.arange(data.x.shape[0])[data.test_mask[:]].tolist()

    elif dataset in ['PPI']:
        graph_index = 19 # There are total 20 graph to use 0~19
        data = PPI('data/PPI/'+str(graph_index))[graph_index]
        adj = to_scipy_sparse_matrix(data.edge_index)
        features = sp.lil_matrix(data.x.numpy())
        labels = (data.y).numpy()
        idx_train = []
        idx_val = []
        for i in range(data.y.shape[1]):
            target = (data.y == i).nonzero(as_tuple=True)[0]
            indices = torch.randperm(len(target))[:20]
            idx_train.extend(target[indices].numpy().tolist())
        
        unselected = torch.ones(data.x.shape[0], dtype=torch.bool)
        unselected[torch.tensor(idx_train)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        indices = torch.randperm(len(target))[:500]
        idx_val.extend(target[indices].numpy().tolist())

        unselected[torch.tensor(idx_val)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        idx_test = target.numpy().tolist()
    
    elif dataset in ['amazon_computers']:
        data = Amazon('data/amazon_computers', 'Computers')[0]
        print(data)
        adj = to_scipy_sparse_matrix(data.edge_index)
        features = sp.lil_matrix(data.x.numpy())
        labels = F.one_hot(data.y).numpy()

        idx_train = []
        idx_val = []
        for i in range(int(max(data.y))+1):
            target = (data.y == i).nonzero(as_tuple=True)[0]
            indices = torch.randperm(len(target))[:20]
            idx_train.extend(target[indices].numpy().tolist())
        
        unselected = torch.ones(data.x.shape[0], dtype=torch.bool)
        unselected[torch.tensor(idx_train)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        indices = torch.randperm(len(target))[:500]
        idx_val.extend(target[indices].numpy().tolist())

        unselected[torch.tensor(idx_val)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        idx_test = target.numpy().tolist()

    elif dataset in ['Cora_ML']:
        data = CitationFull('data/Cora_ML', 'Cora_ML')[0]
        print(data)
        adj = to_scipy_sparse_matrix(data.edge_index)
        features = sp.lil_matrix(data.x.numpy())
        labels = F.one_hot(data.y).numpy()

        idx_train = []
        idx_val = []
        for i in range(int(max(data.y))+1):
            target = (data.y == i).nonzero(as_tuple=True)[0]
            indices = torch.randperm(len(target))[:20]
            idx_train.extend(target[indices].numpy().tolist())
        
        unselected = torch.ones(data.x.shape[0], dtype=torch.bool)
        unselected[torch.tensor(idx_train)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        indices = torch.randperm(len(target))[:500]
        idx_val.extend(target[indices].numpy().tolist())

        unselected[torch.tensor(idx_val)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        idx_test = target.numpy().tolist()

    elif dataset in ['IMDB']:
        data = IMDB('data/IMDB')[0]
        print(data)
        adj = to_scipy_sparse_matrix(data.edge_index)
        features = sp.lil_matrix(data.x.numpy())
        labels = F.one_hot(data.y).numpy()
        idx_train = torch.arange(data.x.shape[0])[data.train_mask[:, 0]].tolist()
        idx_val = torch.arange(data.x.shape[0])[data.val_mask[:, 0]].tolist()
        idx_test = torch.arange(data.x.shape[0])[data.test_mask[:]].tolist()

    elif dataset in ['Twitch']:
        # DE: 9498 nodes
        # EN: 7126 nodes
        # ES: 4648 nodes
        # FR: 6551 nodes
        # PT: 1912 nodes
        # RU: 4385 nodes
        name = 'ES'
        data = Twitch('data/Twitch', name)[0]
        print(data)
        adj = to_scipy_sparse_matrix(data.edge_index)
        features = sp.lil_matrix(data.x.numpy())
        labels = F.one_hot(data.y).numpy()
        idx_train = []
        idx_val = []
        for i in range(int(max(data.y))+1):
            target = (data.y == i).nonzero(as_tuple=True)[0]
            indices = torch.randperm(len(target))[:20]
            idx_train.extend(target[indices].numpy().tolist())

        unselected = torch.ones(data.x.shape[0], dtype=torch.bool)
        unselected[torch.tensor(idx_train)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        indices = torch.randperm(len(target))[:500]
        idx_val.extend(target[indices].numpy().tolist())

        unselected[torch.tensor(idx_val)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        idx_test = target.numpy().tolist()

    elif dataset in ['LastFMAsia']:
        data = LastFMAsia('data/LastFMAsia')[0]
        print(data)
        adj = to_scipy_sparse_matrix(data.edge_index)
        features = sp.lil_matrix(data.x.numpy())
        labels = F.one_hot(data.y).numpy()
        idx_train = []
        idx_val = []
        for i in range(int(max(data.y))+1):
            target = (data.y == i).nonzero(as_tuple=True)[0]
            indices = torch.randperm(len(target))[:20]
            idx_train.extend(target[indices].numpy().tolist())

        unselected = torch.ones(data.x.shape[0], dtype=torch.bool)
        unselected[torch.tensor(idx_train)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        indices = torch.randperm(len(target))[:500]
        idx_val.extend(target[indices].numpy().tolist())

        unselected[torch.tensor(idx_val)] = False
        target = unselected.nonzero(as_tuple=True)[0]
        idx_test = target.numpy().tolist()


    elif dataset in ['USAir', 'PB', 'Celegans', 'Power', 'Router', 'Ecoli', 'Yeast', 'NS']:
        data_dir = 'data/wo_attr/' + dataset + '.mat'
        print('Load data from: '+ data_dir)
        import scipy.io as sio
        net = sio.loadmat(data_dir)
        edge_index, _ = from_scipy_sparse_matrix(net['net'])
        node_num = torch.max(edge_index) + 1
        if is_undirected(edge_index) == False:
            edge_index = to_undirected(edge_index)
        adj = to_scipy_sparse_matrix(edge_index)

        features = None
        labels = None
        idx_train = None
        idx_val = None
        idx_test = None

    else:
        print('Not Implemented')
        return None

    return adj, features, labels, idx_train, idx_val, idx_test

if __name__ == '__main__':
    load_data('ogbl-collab')