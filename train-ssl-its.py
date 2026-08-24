# Iterative Teacher-Student implementation

import multiprocessing
import pandas as pd
import numpy as np
import time
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import SGDClassifier
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

import augmentations

label_encoder = LabelEncoder()

class ModelData:
	features: np.ndarray
	labels:   np.ndarray
	X:        np.ndarray # reference for function congruency
	y:        np.ndarray # reference for function congruency
						 # augmentation functions work on X and y
						 # so have them point to the necessary
						 # data structure.
	X_labeled:    np.ndarray # input matrix (features)
	y_labeled:    np.ndarray # input labels
	X_unlabeled:  np.ndarray # 50% of input set as unlabeled
	#y_unlabeled:  np.ndarray # pseudo labels
	#X_unlab_aug: np.ndarray # augmented unlabeled matrix
	#y_unlab_aug: np.ndarray # augmented pseudo labels
	X_test:   np.ndarray # reserved data to test model
	y_test:   np.ndarray # labels for model testing.
	y_pred:   np.ndarray	
	feature_names:   np.ndarray
	acc_per_loop:    np.ndarray
	max_confidence:  np.ndarray
	tau_per_loop:	 np.ndarray
	min_confidence:  np.ndarray
	percent_confident: np.ndarray
	class_list:		 np.ndarray
	noise: float
	tau: float
	name: str
	xcombined_shape: np.ndarray


def load_data(path, md):
	data = pd.read_csv(path)
	md.features = data.iloc[0:, data.columns != "Group"].to_numpy()
	md.feature_names = data.columns[data.columns != "Group"].tolist()
	md.labels = data['Group'].to_numpy()

def preprocess(md):
    # handle zeros
	pseudo_count = 1.0 
	features_safe = md.features + pseudo_count

	# Calculate the geometric mean across features for each sample
	# (row-wise, axis=1)
	geom_mean = np.exp(np.mean(np.log(features_safe), axis=1, keepdims=True))
	
	# Apply the Centered Log-Ratio (CLR)
	md.features = np.log(features_safe / geom_mean)

	# Encode labels if categorical (e.g., 'A', 'B' -> 0, 1) 
	# if labels.dtype == 'str': 
	md.labels = label_encoder.fit_transform(md.labels)

	# initialize these arrays so that they can be appended to
	md.acc_per_loop = np.array([]) 
	md.tau_per_loop = np.array([]) 
	md.max_confidence = np.array([]) 
	md.min_confidence = np.array([]) 
	md.percent_confident = np.array([])
	md.xcombined_shape = np.array([])

	# classes must be the *full* list from the start
	md.class_list = np.unique(md.labels)

	# Split into train/test sets 
	# First split: 80% data, 20% test (unlabeled)
	X_train_full, md.X_test, y_train_full, md.y_test = train_test_split(
			md.features, md.labels,
			test_size=0.2,
		   	random_state=42, stratify=md.labels
	)

	# We can delete this data, it is no longer needed
	del md.features, md.labels

	# Second split: 50% train, 50% unlabeled
	md.X_labeled, md.X_unlabeled, md.y_labeled, _ = train_test_split(
			X_train_full, y_train_full, test_size=0.5,
		   	random_state=42, stratify=y_train_full)

def batch_weights(y_batch, class_list):
    weights = compute_class_weight('balanced', classes=class_list, y=y_batch)
    return np.array([dict(zip(class_list, weights))[y] for y in y_batch])


#@profile
def trainingLoop(md):
	md.name = "SDGClassifier"
	scaler = StandardScaler()
	md.X_labeled = scaler.fit_transform(md.X_labeled)

	# Try to augment first and then transform
	#md.X_unlabeled = scaler.transform(md.X_unlabeled)
	
	md.X_test = scaler.transform(md.X_test)

	# Initialise incremental model
	clf = SGDClassifier(loss='log_loss', max_iter=10, warm_start=True, 
		learning_rate='optimal', random_state=42)

	# Fit once on the initial labeled set
	clf.partial_fit(md.X_labeled, md.y_labeled, classes=md.class_list)

	md.y_pred = clf.predict(md.X_test)
	acc = accuracy_score(md.y_test, md.y_pred)
	print(f"*Base Accuracy*: {acc:.4f}")

	# Keep a list of indices of samples that have been “used”
	used = np.zeros(len(md.X_unlabeled), dtype=bool)
	for loop in range(md.totalLoops):
		md.X = md.X_unlabeled[~used]
		print(f"X size: {md.X.shape}")

		#augmentations.aitchisonPerturbation(md, noise_scale=md.noise)
		augmentations.augmentTabular1(md, noise_std=md.noise)
		#augmentations.augmentPassthru(md, noise_std=md.noise)
		md.X_augmented = scaler.transform(md.X_augmented)
		print(f"X-unlab size: {md.X_unlabeled.shape}")
		print(f"X-aug size: {md.X_augmented.shape}")

		# Predict on unlabeled data. predict_prob rows sum to 1
		pseudo_unlab = clf.predict_proba(md.X_augmented)

		# Convert probability predictions into class labels
		md.y_augmented = np.argmax(pseudo_unlab, axis=1)

		confidences = np.max(pseudo_unlab, axis=1)
		md.percent_confident = np.append(
			md.percent_confident, np.mean(confidences >= md.tau)) # * 100 

		# Filter high-confidence pseudo-labels
		mask = confidences >= md.tau
		print("\tHigh conf size:" + str(mask.shape[0]))
		print("\tHigh conf %:" + str(md.percent_confident))

		if not mask.any():
			print(f"Loop {loop}: no confident samples, stopping.")
			break

		indexes = np.where(mask)[0]  # Returns array of indexes
		print("\t size:" + str(indexes.shape[0]))
		used[indexes] = True

		md.X_labeled = np.concatenate([md.X_labeled, md.X_augmented[mask]])
		md.y_labeled = np.concatenate([md.y_labeled, md.y_augmented[mask]])

		# Update the model with pseudo‑labels
		sample_weights = batch_weights(md.y_labeled, md.class_list)
		clf.partial_fit(md.X_augmented[mask], md.y_augmented[mask], classes=md.class_list, sample_weight=sample_weights)

		# Evaluate 
		md.y_pred = clf.predict(md.X_test)
		acc = accuracy_score(md.y_test, md.y_pred)

		md.acc_per_loop = np.append(md.acc_per_loop, acc)
		print(f"\tLoop: {loop} \t Accuracy: {acc:.4f}")

	return clf


def createPlot(md):
	font = {'family': 'serif', 'size': 8}

	plt.clf()
	plt.plot(md.acc_per_loop,   "g-", linewidth=1, label="Accuracy")
	plt.plot(md.percent_confident, "y-", linewidth=1,
		label="Percent Confident")
	
	plt.xlabel("Iteration")
	plt.ylabel("Accuracy/Confidence %")
	plt.title("Per iteration Tau =" + str(md.tau) + " - Noise = " + str(md.noise))
	plt.legend(loc='best')
	plt.grid(True, alpha=0.3)
	# from 0 to total loop in steps of ....
	plt.xticks(np.arange(0, md.totalLoops + 1,
		(md.totalLoops if md.totalLoops < 10 else md.totalLoops / 10)))
	# Save the plot to an image file
	now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
	plt.savefig("rf-T" + str(md.tau) + "-N" + str(md.noise) + "-" + now + ".png", dpi=300, bbox_inches='tight')

def createDebugPlot(md):
	font = {'family': 'serif', 'size': 8}

	plt.clf()
	plt.plot(md.xcombined_shape, "r-", linewidth=1, label="Trained")
	
	plt.xlabel("Iteration")
	plt.ylabel("Train Data Size")
	plt.title("Training Data Growth Tau =" + str(md.tau) + " - Noise = " + str(md.noise))
	plt.legend(loc='best')
	plt.grid(True, alpha=0.3)
	# from 0 to total loop in steps of ....
	plt.xticks(np.arange(0, md.totalLoops + 1,
		(md.totalLoops if md.totalLoops < 20 else md.totalLoops / 10)))
	# Save the plot to an image file
	now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
	plt.savefig("rf-T" + str(md.tau) + "-N" + str(md.noise) + "-growth-" + now + ".png", dpi=300, bbox_inches='tight')


def displayMetrics(md):
	print(md.name)

	# Classification metrics
	print(f"Accuracy: {accuracy_score(md.y_test, md.y_pred):.2f}")

	createConfusionMatrix(md)
	
	# Regression metrics 
	# print(f"RMSE: {mean_squared_error(md.y_test, md.y_pred, squared=False):.2f}")
	

def createConfusionMatrix(md):
	cm = confusion_matrix(md.y_test, md.y_pred)
	plt.figure(figsize=(10, 6))

	sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
			xticklabels=label_encoder.classes_,
			yticklabels=label_encoder.classes_)
	
	plt.title("Confusion Matrix")
	plt.xlabel("Predicted")
	plt.ylabel("True")
	starttime = datetime.now().strftime("%H%M%S")
	plt.savefig(md.name + ".cm." + starttime + ".png")


def rfFeatureImportance(md, classifier):
	featImp = pd.DataFrame({
		"Feature": md.feature_names,
		"Importance": classifier.feature_importances_,
	}).sort_values("Importance", ascending=False)

	print("\n=== Top Random Forest Important Features ===")
	print(featImp.head(10))

	# Plot feature importance
	plt.figure(figsize=(10, 6))
	sns.barplot(x="Importance", y="Feature", data=featImp.head(10))
	plt.title("Top 10 Important Features")

	# Save the plot to an image file
	starttime = datetime.now().strftime("%Y-%m-%d-%H%M%S")
	plt.savefig(md.name + ".featimp." + starttime + ".png", dpi=300, bbox_inches='tight')

def outputArray(arr, filename, fmtstr='%d'):
	now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
	np.savetxt(filename + "-" + now + ".tsv", arr, delimiter='\t', fmt=fmtstr)
	time.sleep(2)

def printTime(msg):
	starttime = datetime.now().strftime("%Y-%m-%d-%H%M%S")
	print(msg + ": " + starttime)

def main():
	infile = ""

	if len(sys.argv) > 1:
		infile = sys.argv[1]
	else:
		print("File name ommited");
		return

	printTime("Process Start")
	md = ModelData()
	load_data(infile, md)
	preprocess(md)

	#printTime("Random Forest start")
    
	md.tau = .95
	md.noise = 0.1
	md.totalLoops = 14
	rf = trainingLoop(md)
	#createPlot(md)
	#createDebugPlot(md)

	# rfFeatureImportance(md, rf)
	# displayMetrics(md)
	#printTime("Random Forest End")

	printTime("Process End")


main()


