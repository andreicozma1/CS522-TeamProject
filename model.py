import pickle
import os


class Model:
    def __init__(self, model, y, cm):
        self.model = model
        self.y = y
        self.cm = cm

    def save(self, filename, dir_path='models'):
        with open(os.path.join(dir_path, filename), 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename, dir_path='models'):
        with open(os.path.join(dir_path, filename), 'rb') as f:
            return pickle.load(f)
