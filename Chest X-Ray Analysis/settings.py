class Settings():
    def __init__(self):
        self.labels =  ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']
        self.IMAGE_DIR = "nih/images-small/"
        self.train_df_path = "nih/train-small.csv"
        self.valid_df_path = "nih/valid-small.csv"
        self.test_df_path = "nih/test.csv"
