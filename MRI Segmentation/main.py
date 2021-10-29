from preprocess import load_case, load_data, result, visualize_result
import argparse
from metrics import compute_metrics, get_sens_spec_df

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input file of images")
ap.add_argument("-e", "--label", required=True,
	help="path to file of labels")
args = vars(ap.parse_args())

image_file = args["image"]
label_file = args["label"]

X, y = load_data(image_file, label_file)

pred = result(X)

metrics = get_sens_spec_df(pred[0],y)
print(metrics)

visualize_result(X,pred,y)
