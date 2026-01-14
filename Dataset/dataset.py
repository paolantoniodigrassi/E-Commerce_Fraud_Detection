import pandas as pd

class Dataset:

    def __init__(self, path):
        self.dataset = pd.read_csv(path)
    
    def getDataset(self):
        return self.dataset
    
    def setDataset(self, dataset):
        self.dataset = dataset

    def dropDatasetColumns(self, columnsToRemove):
        self.dataset = self.dataset.drop(columns=columnsToRemove)
        
    def saveDataset(self, path):
        self.dataset.to_csv(path, index=False)