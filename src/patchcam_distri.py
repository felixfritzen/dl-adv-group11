from collections import Counter
import datasets.datasets as datasets
import torch
def get_label_distribution(dataloader):
    label_counts = Counter()
    for _, labels in dataloader:
        label_counts.update(labels.tolist())
    return label_counts


def test():
    dataloaders = datasets.get_dataloaders('camelyon')

    train_indices = {idx for idx, _, _ in dataloaders['train']}
    val_indices = {idx for idx, _, _ in dataloaders['val']}
    overlap = train_indices & val_indices
    print(f"Overlap: {len(overlap)}")  # Should be 0

    train_label_counts = get_label_distribution(dataloaders['val'])
    print(f"Training Label Distribution: {train_label_counts}")

dataloaders = datasets.get_dataloaders('camelyon')

all_labels = []
print(len(dataloaders['train']))
for _, label in dataloaders['train']:
    all_labels.extend(label.numpy() if torch.is_tensor(label) else label)

# Count the occurrences of each label
label_counts = Counter(all_labels)
print(label_counts)