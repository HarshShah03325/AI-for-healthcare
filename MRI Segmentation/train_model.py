import model
import util
import json

def train_model(steps_per_epoch, n_epochs, validation_steps):
    
    base_dir = 'BraTS-Data/processed'
    with open("config.json") as json_file:
        config = json.load(json_file)

    # Get generators for training and validation sets
    train_generator = util.VolumeDataGenerator(config["train"], base_dir + "train/", batch_size=3, dim=(160, 160, 16), verbose=1)
    valid_generator = util.VolumeDataGenerator(config["valid"], base_dir + "valid/", batch_size=3, dim=(160, 160, 16), verbose=1)

    trained_model = model.Unet()

    trained_model.fit_generator(generator=train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=n_epochs,
        use_multiprocessing=True,
        validation_data=valid_generator,
        validation_steps=validation_steps)
    
    return trained_model


