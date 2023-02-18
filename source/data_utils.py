import torch
from torch_geometric.data import Dataset, InMemoryDataset, Data
import os
from tqdm import tqdm

from torch_geometric.utils import dense_to_sparse


class NYCCab(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root
        self.name = 'nyc_cab'
        self.cleaned = False
        self.graph_file = 'graphs.pt'
        self.event_file = 'events.pt'
        self.event_types_file = 'event_types.pt'
        self.static_features_file = 'static_attrs.pt'
        self.dynamic_features_file = 'dynamic_attrs.pt'
        self.graph_count = 4464
        super(NYCCab, self).__init__(root, transform, pre_transform)

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return [self.graph_file, self.event_file, self.event_types_file, self.static_features_file, self.dynamic_features_file]

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        return [f'data_{i}.pt' for i in range(self.graph_count)]

    @property
    def num_classes(self) -> int:
        r"""Returns the number of classes in the dataset."""
        return 2

    def download(self):
        pass

    def process(self):
        graphs = torch.load(os.path.join(self.raw_dir, self.graph_file))
        labels = torch.load(os.path.join(self.raw_dir, self.event_file))
        label_types = torch.load(os.path.join(self.raw_dir, self.event_types_file))
        static_features = torch.load(os.path.join(self.raw_dir, self.static_features_file))
        dynamic_features = torch.load(os.path.join(self.raw_dir, self.dynamic_features_file))

        # TODO: please normalize features or graphs if it is necessary.

        for i, g in tqdm(enumerate(graphs)):
            edge_index, edge_weights = dense_to_sparse(g)
            d = Data(edge_index=edge_index.clone(),
                     edge_weight=edge_weights.clone(),
                     x=torch.cat([static_features.clone(), dynamic_features[i].clone()], dim=1).float(),
                     y=labels[i].clone().long(),
                     # -1 for 0 labels (no event)
                     y_type=label_types[i].clone().long())
            torch.save(d, os.path.join(self.processed_dir, f'data_{i}.pt'))

    def len(self):
        return self.graph_count

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{idx}.pt'))
        return data


class TwitterWeather(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root
        self.name = 'twitter_weather'
        self.cleaned = False
        self.graph_file = 'graphs.pt'
        self.event_file = 'events.pt'
        self.event_types_file = 'event_types.pt'
        self.static_features_file = 'static_attrs.pt'
        self.dynamic_features_file = 'dynamic_attrs.pt'
        self.graph_count = 2557
        super(TwitterWeather, self).__init__(root, transform, pre_transform)

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return [self.graph_file, self.event_file, self.event_types_file, self.static_features_file, self.dynamic_features_file]

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        return [f'data_{i}.pt' for i in range(self.graph_count)]

    @property
    def num_classes(self) -> int:
        r"""Returns the number of classes in the dataset."""
        return 2

    def download(self):
        pass

    def process(self):
        graphs = torch.load(os.path.join(self.raw_dir, self.graph_file))
        labels = torch.load(os.path.join(self.raw_dir, self.event_file))
        label_types = torch.load(os.path.join(self.raw_dir, self.event_types_file))
        static_features = torch.load(os.path.join(self.raw_dir, self.static_features_file))
        dynamic_features = torch.load(os.path.join(self.raw_dir, self.dynamic_features_file))

        # TODO: please normalize features or graphs if it is necessary.

        for i, g in tqdm(enumerate(graphs)):
            edge_index, edge_weights = dense_to_sparse(g)
            d = Data(edge_index=edge_index.clone(),
                     edge_weight=edge_weights.clone(),
                     x=torch.cat([static_features.clone(), dynamic_features[i].clone()], dim=1).float(),
                     y=labels[i].clone().long(),
                     # -1 for 0 labels (no event)
                     y_type=label_types[i].clone().long())
            torch.save(d, os.path.join(self.processed_dir, f'data_{i}.pt'))

    def len(self):
        return self.graph_count

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{idx}.pt'))
        return data


def load_dataset(dataset_name):
    if dataset_name == 'nyc_cab':
        data = NYCCab(root='data/')
    elif dataset_name == 'twitter_weather':
        data = TwitterWeather(root='data/')
    else:
        raise NotImplementedError(f'Dataset: {dataset_name} is not implemented!')

    return data


if __name__ == '__main__':
    dataset = load_dataset('twitter_weather')
    print(f'Size: {len(dataset)}')
    print(f'Labels sum: {sum([graph.y for graph in dataset]) / len(dataset)}')
