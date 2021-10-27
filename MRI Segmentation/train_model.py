import model
import json
from util import VolumeDataGenerator

def train_model(steps_per_epoch, n_epochs, validation_steps):
    '''
    trains a 3D Unet for give parameters:
    steps_per_epoch
    n_epochs
    validation_steps
    The function uses train generator and valid generator for model.fit_generator() method.
    '''
    
    base_dir = 'BraTS-Data/processed/'
    with open("config.json") as json_file:
        config = json.load(json_file)

    # Get generators for training and validation sets
    train_generator = VolumeDataGenerator(config["train"], base_dir + "train/", batch_size=3, dim=(160, 160, 16), verbose=1)
    valid_generator = VolumeDataGenerator(config["valid"], base_dir + "valid/", batch_size=3, dim=(160, 160, 16), verbose=1)

    trained_model = model.Unet()

    trained_model.fit_generator(generator=train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=n_epochs,
        use_multiprocessing=False,
        validation_data=valid_generator,
        validation_steps=validation_steps)
    
    return trained_model

def save_model(steps_per_epoch=2, n_epochs=1, validation_steps=2):
    '''
    Save weights of a model trained for given parameters:
    steps_per_epoch
    n_epochs
    validation_steps
    '''

    model = train_model(steps_per_epoch, n_epochs,validation_steps)
    model.save_weights('pretrained_model.hdf5')

save_model()