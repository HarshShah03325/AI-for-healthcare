import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import nibabel as nib
import tensorflow as tf
from keras import backend as K
from model import load_model
from util import visualize_patch

# HOME_DIR = "./BraTS-Data/"
# DATA_DIR = HOME_DIR
# base_dir = HOME_DIR + "processed/"

def single_class_dice_coefficient(y_true, y_pred, axis=(0, 1, 2), epsilon=0.00001):
    ''' 
    Compute dice coefficient for single class.
    returns:
        dice_coefficient(float): computed value of dice coefficient. 
    '''
    dice_numerator = 2 * np.sum(y_true * y_pred, axis = axis) + epsilon
    dice_denominator = K.sum(y_true,axis= axis) + K.sum(y_pred, axis= axis) + epsilon
    dice_coefficient = dice_numerator/dice_denominator
    return dice_coefficient


def soft_dice_loss(y_true, y_pred, axis=(1, 2, 3), epsilon=0.00001):
    '''
    Compute mean dice coefficient over all abnormality classes.
    returns:
        dice_coefficient (float): computed value of dice coefficient.
    '''
    dice_numerator = 2 * K.sum(y_true * y_pred , axis= axis) + epsilon
    dice_denominator = K.sum(y_true ** 2, axis= axis) + K.sum(y_pred ** 2 , axis = axis) + epsilon
    dice_loss = 1 - K.mean(dice_numerator / dice_denominator)
    return dice_loss


def dice_coefficient(y_true, y_pred, axis=(1, 2, 3), epsilon=0.00001):
    '''
    Compute mean soft dice loss over all abnormality classes.
    returns:
        dice_loss (float): computed value of dice loss.  
    '''
    dice_numerator = 2 * K.sum(y_true * y_pred , axis = axis) + epsilon
    dice_denominator = K.sum(y_true, axis = axis ) + K.sum(y_pred, axis = axis) + epsilon
    dice_coefficient = K.mean(dice_numerator/dice_denominator)
    return dice_coefficient


def load_case(image_nifty_file, label_nifty_file):
    '''load the image and label file, get the image content and return a numpy array.
    returns:
        image(np.array): image from input file.
        label(np.array) label from input file.
    '''
    image = np.array(nib.load(image_nifty_file).get_fdata())
    label = np.array(nib.load(label_nifty_file).get_fdata())
    return image, label


def get_sub_volume(image, label, 
                   orig_x = 240, orig_y = 240, orig_z = 155, 
                   output_x = 160, output_y = 160, output_z = 16,
                   num_classes = 4, max_tries = 1000, 
                   background_threshold=0.95):
    ''' Extract random sub volume from original images.
    returns:
        X (np.array): sample of original image of dimension 
            (num_channels, output_x, output_y, output_z).
        y (np.array): labels which correspond to X, of dimension 
            (num_classes, output_x, output_y, output_z).
    '''
    X = None
    y = None
    tries = 0
    
    while tries < max_tries:
        start_x = np.random.randint(0,orig_x-output_x+1)
        start_y = np.random.randint(0,orig_y-output_y+1)
        start_z = np.random.randint(0,orig_z-output_z+1)

        y = label[start_x: start_x + output_x,
                  start_y: start_y + output_y,
                  start_z: start_z + output_z]
        
        y = tf.keras.utils.to_categorical(y, num_classes = num_classes)

        bgrd_ratio = np.sum(y[:,:,:,0]) / (output_x * output_y * output_z)

        tries += 1

        if bgrd_ratio < background_threshold:

            X = np.copy(image[start_x: start_x + output_x,
                              start_y: start_y + output_y,
                              start_z: start_z + output_z, :])
            
            X = np.moveaxis(X,3,0)
            y = np.moveaxis(y,3,0)

            y = y[1:, :, :, :]
    
            return X, y

    print(f"Tried {tries} times to find a sub-volume. Giving up...")
    exit


def standardize(image):
    '''
    Standardize mean and standard deviation of each channel and z_dimension.
    returns:
        standardized version of input image(np.array).
    '''
    
    standardized_image = np.zeros_like(image)

    # iterate over channels
    for c in range(image.shape[0]):
        # iterate over the `z` dimension
        for z in range(image.shape[3]):
            # get a slice of the image 
            # at channel c and z-th dimension `z`
            image_slice = image[c,:,:,z]

            # subtract the mean from image_slice
            centered = image_slice - np.mean(image_slice)
            
            # divide by the standard deviation (only if it is different from zero)
            if np.std(centered) != 0:
                centered_scaled = centered/np.std(centered)

                # update  the slice of standardized image
                # with the scaled centered and scaled image
                standardized_image[c, :, :, z] = centered_scaled

    return standardized_image


def load_data(image_file,label_file):
    '''
    Loads images and labels from input files.
    Standardizes the image, swaps channel dimension with depth, adds one dimension(images_per_batch).
    returns:
        X_norm_with_batch_dimension(np.array) : processed version of the image.
        y(np.array) : label image.
    '''
    image, label = load_case(image_file,label_file)
    X, y = get_sub_volume(image, label)
    X_norm = standardize(X)
    X_norm = np.swapaxes(X_norm,0,3)
    X_norm_with_batch_dimension = np.expand_dims(X_norm, axis=0)
    return X_norm_with_batch_dimension,y


def result(X,threshold=0.5):
    '''
    Loads a pre-trained 3D-UNet model.
    Predicts the output for input image(np.array) and applies and threshold.
    returns:
        patch_pred(np.array) predicted image.
    '''
    mod = load_model()
    patch_pred = mod.predict(X)
    patch_pred = np.swapaxes(patch_pred,1,4)
    patch_pred[patch_pred > threshold] = 1.0
    patch_pred[patch_pred <= threshold] = 0.0
    return patch_pred

def visualize_result(X_norm,patch_pred,y):
    '''
    Plots the image with ground truth and predicted image.
    '''
    # print("Patch and ground truth")
    # visualize_patch(X_norm[0, :, :, :], y[2])
    # plt.show()
    print("Patch and prediction")
    visualize_patch(X_norm[0, :, :, :], patch_pred[0, 2, :, :, :])
    plt.show()






















