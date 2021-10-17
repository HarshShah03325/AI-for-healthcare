from preprocess import load_data, result, visualize_result
import argparse

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

visualize_result(X,pred,y)
