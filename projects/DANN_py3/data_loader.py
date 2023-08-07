from  torch.utils.data import DataLoader, SubsetRandomSampler, random_split 
from torchvision.datasets import ImageFolder

def create_dataloader(dataset,transform, batch_size ,train_ratio, validation_ratio):

    data_source = ImageFolder(root=dataset, transform=transform)
    num_examples = len(data_source)
    train_size = int(train_ratio * num_examples)
    validation_size = int(validation_ratio * num_examples)
    test_size = num_examples - train_size - validation_size

    train_indices, remaining_indices = random_split(range(num_examples), [train_size, num_examples - train_size])
    validation_indices, test_indices = random_split(remaining_indices, [validation_size, test_size])

    train_loader = DataLoader(data_source, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices), num_workers=2)
    validation_loader = DataLoader(data_source, batch_size=batch_size, sampler=SubsetRandomSampler(validation_indices), num_workers=2)
    test_loader = DataLoader(data_source, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices), num_workers=2)

    return train_loader, validation_loader, test_loader
