import pandas as pd
import sys
from sklearn.preprocessing import normalize
 
def normalizeData(input_csv, output_csv):
	# Read the CSV file 
	inData = pd.read_csv(input_csv)
	labels = inData.iloc[:, 0]
	features = inData.iloc[:, 1:]

	outData = pd.DataFrame(normalize(features, axis=1, norm="l1"))
	outData.insert(0, inData.columns[0], labels);

	# Save the normalized data to a new CSV file
	outData.to_csv(output_csv, index=False)
	print(f"Normalized data saved to {output_csv}")

def main():
	if len(sys.argv) > 1:
		infile = sys.argv[1]
		outfile = sys.argv[1] + ".output"
		normalizeData(infile, outfile);

main()

