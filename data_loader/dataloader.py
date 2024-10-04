import logging
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd


class Test(Dataset):
    """
    Class to preprocess the dataset.
    """

    def __init__(self) -> None:
        super().__init__()
        logging.debug(
            "\n" + "\n" + " ###################### " + \
            "\n" + " #### get raw data #### " + \
            "\n" + " ###################### " + "\n"
        )
        
        df = pd.read_csv("./test_data.csv", on_bad_lines="skip")

        logging.debug(
            "\n" + "\n" + " ######################## " + \
            "\n" + " #### run preprocess #### " + \
            "\n" + " ######################## " + "\n"
        )

        df["inputs"] = df["inputs"].str.replace('[', '', regex=False)
        df["inputs"] = df["inputs"].str.replace(']', '', regex=False)
       
        df["labels"] = df["labels"].str.replace('[', '', regex=False)
        df["labels"] = df["labels"].str.replace(']', '', regex=False)
        
        self.questions = df["inputs"].str.split(',', expand=True)
        self.responses = df["labels"].str.split(',', expand=True)
        
        self.questions.index = range(0, len(self.questions))
        self.responses.index = range(0, len(self.responses))

        self.length = len(self.questions)

    def __getitem__(self, index):
        question = np.array(self.questions.loc[index]).astype(np.float32)
        response = np.array(self.responses.loc[index]).astype(np.float32)
        
        return torch.tensor(question, dtype=torch.float32), torch.tensor(response, dtype=torch.float32)

    def __len__(self):
        return self.length
