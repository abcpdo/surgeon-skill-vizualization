from torch.utils.data import Dataset, DataLoader


class JigsawsDataset(Dataset):
    def __init__(self, csv_novice, csv_expert):
        self.Tensor = generator_tensor(csv_novice, csv_expert)

    def generator_tensor():

        return Tensor

    def __len__(self):
        return len(self)
