from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from keras import backend as K
from settings import Settings
import numpy as np
from helper import get_train_labels, compute_class_freqs
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

settings = Settings()


def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """
    Custom loss function that calculates loss based on positive and negative weights.
    Weights are inversely proportional to frequencies.
    returns: 
        weighted loss.
    """
    def weighted_loss(y_true, y_pred):
        
        loss = 0.0
        
        for i in range(len(pos_weights)): 
            loss += -1 * K.mean(pos_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon)) + -1 * K.mean(neg_weights[i] * (1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon)) #complete this line
        return loss
    
    return weighted_loss



def denseNet():
    """
    Builds and compiles the keras DenseNet model.
    returns:
        Untrained DenseNet model.
    """
    base_model = DenseNet121(weights='./densenet.hdf5', include_top=False)
    
    x = base_model.output

    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)

    # and a logistic layer
    predictions = Dense(len(settings.labels), activation="sigmoid")(x)

    pos_weights, neg_weights = compute_class_freqs(get_train_labels())
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights), experimental_run_tf_function=False)
    
    return model


def load_model():
    """
    Builds keras DenseNet model and loads pretrained weights into the model.
    returns:
        Trained DenseNet model.
    """
    model = denseNet()
    model.load_weights("./pretrained_model.h5")
    return model

