from model import load_model
from settings import Settings
from util import get_roc_curve, compute_gradcam
import numpy as np
from helper import get_df, load_generator

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input file of images")
args = vars(ap.parse_args())


settings = Settings()
mod = load_model()

train_df = get_df(settings.train_df_path)
valid_df = get_df(settings.valid_df_path)
test_df = get_df(settings.test_df_path)

train_generator, valid_generator, test_generator = load_generator(train_df, valid_df, test_df)
predicted_vals = mod.predict_generator(test_generator, steps = len(test_generator))

auc_rocs = get_roc_curve(settings.labels, predicted_vals, test_generator)


labels_to_show = np.take(settings.labels, np.argsort(auc_rocs)[::-1])[:4]
compute_gradcam(mod, args["image"], '', train_df, settings.labels, labels_to_show)