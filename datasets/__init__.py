from .tea import Tea


dataset_list = {
    "tea": Tea,
}


def build_dataset(dataset, root_path, shots):
    return dataset_list[dataset](root_path, shots)