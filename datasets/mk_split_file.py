from torch.utils.data import Dataset
import os
import cv2


class Data(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.dir_list_dir = os.listdir(data_dir)
        self.img_list_dir = []
        for i in range(len(self.dir_list_dir)):
            self.img_list_dir.append(os.listdir(os.path.join(data_dir, self.dir_list_dir[i])))

    def __getitem__(self, index1, index2):
        img_name = self.img_list_dir[index1][index2]
        img_item_path = os.path.join(self.data_dir, self.dir_list_dir[index1], img_name)
        img = cv2.imread(img_item_path)
        return img
    
    def __dirlen__(self):
        return len(self.dir_list_dir)
    
    def __imglen__(self, index):
        return len(self.img_list_dir[index])
    
    
def main():
    data_dir = 'original_data'
    dataset = Data(data_dir)
    
    with open('trainval.txt', 'w') as f:
        for i in range(dataset.__dirlen__()):
            for j in range(dataset.__imglen__(i)):
                f.write(f"{dataset.img_list_dir[i][j].split('.')[0]} {i + 1}\n")
                
if __name__ == "__main__":
    main()