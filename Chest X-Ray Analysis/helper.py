from tensorflow.keras.preprocessing.image import ImageDataGenerator
from settings import Settings
import numpy as np
import pandas as pd

settings = Settings()

def check_for_leakage(df1, df2, patient_col):
    """
    Checks for same patients in both samples.
    """
    df1_patients_unique = set(df1[patient_col].values)
    df2_patients_unique = set(df2[patient_col].values)
    patients_in_both_groups = df1_patients_unique.intersection(df2_patients_unique)
    leakage = len(patients_in_both_groups) > 0 
    
    return leakage


def get_train_generator(df, image_dir, x_col, y_cols, shuffle=True, batch_size=8, seed=1, target_w = 320, target_h = 320):
    """
    Computes train generator using traning samples.
    """
    image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization= True) 
    
    generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            target_size=(target_w,target_h))
    
    return generator


def get_test_and_valid_generator(valid_df, test_df, train_df, image_dir, x_col, y_cols, sample_size=100, batch_size=8, seed=1, target_w = 320, target_h = 320):
    """
    Computes generator for testing and validation using inbuilt keras function.
    """
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df, 
        directory=settings.IMAGE_DIR,  #Path of image directory from settings class
        x_col="Image", 
        y_col=settings.labels, #labels from defined settings class 
        class_mode="raw", 
        batch_size=sample_size, 
        shuffle=True, 
        target_size=(target_w, target_h))
    
    batch = raw_train_generator.next()
    data_sample = batch[0]

    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization= True)
    
    image_generator.fit(data_sample)

    valid_generator = image_generator.flow_from_dataframe(
            dataframe=valid_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))

    test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
    
    return valid_generator, test_generator


def get_df(path):
    """ Reads a csv file and returns a pandas dataframe."""
    df = pd.read_csv(path)
    return df


def get_train_labels():
    """ Return labels of train generator. """
    train_generator = get_train_generator(get_df(settings.train_df_path),settings.IMAGE_DIR,"Image",settings.labels)
    return train_generator.labels


def load_generator(train_df,valid_df,test_df):
    """
    Load train, test and validation generator and return it.
    """
    train_generator = get_train_generator(train_df, settings.IMAGE_DIR, "Image", settings.labels)
    valid_generator, test_generator= get_test_and_valid_generator(valid_df, test_df, train_df, settings.IMAGE_DIR, "Image", settings.labels)
    
    return train_generator, valid_generator, test_generator


def compute_class_freqs(labels):
    """
    Computes number of positive and negative frequencies from the labels.
    """
    N = labels.shape[0]
    positive_frequencies = np.sum(labels, axis=0)/N
    negative_frequencies = 1 - positive_frequencies

    return positive_frequencies, negative_frequencies
