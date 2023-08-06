from  torch.utils.data import DataLoader,SubsetRandomSampler

from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder

def create_dataloader(dataset,transform, batch_size ,train_ratio):

    data_source = ImageFolder(root=dataset, transform=transform)
    num_examples_source = len(data_source)
    indices = list(range(num_examples_source))

    train_indices, test_indices = train_test_split(indices, train_size=train_ratio, random_state=42)

    # Create data samplers for train and test sets
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Create data samplers for train, validation, and test sets
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Create the data loaders
    train_loader = DataLoader(data_source, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(data_source, batch_size=batch_size, sampler=test_sampler)

    return train_loader, test_loader