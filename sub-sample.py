# Create sub-samples of a dataset
#	- by percent of each label
#	- by number of samples from each label

import pandas as pd
import random
import sys
import os
import datetime

def stratify_samples_by_count(dataframe, count: int):
	class_column = dataframe.columns[0]
	sampled_dfs = []

	# Identify unique classes
	unique_classes = dataframe[class_column].unique()

	print(f"Found {len(unique_classes)} unique classes.")

	# Group, sample, and collect the results
	for class_label in unique_classes:
		# Filter the DataFrame for the current class
		class_df = dataframe[dataframe[class_column] == class_label]

		# Sample up to max_count rows (using .head() for simplicity)
		sample_df = class_df.head(count)

		sampled_dfs.append(sample_df)

	# Combine all sampled data into one DataFrame
	final_df = pd.concat(sampled_dfs, ignore_index=True)
	print(f"Output rows saved: {len(final_df)} (Max sample size per class: {count})")

	return final_df

def stratify_samples_by_percent (dataframe, ratio: float = 0.1):
	# Class column is the first column (index 0)
	class_column = dataframe.columns[0]

	# Group the DataFrame by the class column
	grouped = dataframe.groupby(class_column)

	sampled_data_list = []

	# Iterate through each unique class and sample 10%
	for class_name, group in grouped:
		# Calculate the number of samples to take
		# Ensure at least 1 sample if possible
		num_samples = max(1, int(len(group) * ratio))

		# Randomly select the desired number of rows from this group
		sampled_group = group.sample(n=min(num_samples, len(group)), random_state=42)

		print(f"Class '{class_name}': Found {len(group)} samples, keeping {len(sampled_group)}.")

		# Append the sampled group to the list
		sampled_data_list.append(sampled_group)

	# Combine all sampled groups into a single DataFrame
	return pd.concat(sampled_data_list).reset_index(drop=True)

def randomize(df):
	return df.sample(frac=1)


def main():
	if len(sys.argv) > 1:
		now = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
		infile = sys.argv[1]
		outfile = infile + "-" + now + ".csv"
		
		df = pd.read_csv(infile)

		if df.empty:
			print("Error in CSV file: it is empty.")
			return

		# df_out = stratify_samples_by_percent(df, ratio = .05);
		df_out = stratify_samples_by_count(df, count = 10);
		df_out = randomize(df_out)
		df_out.to_csv(outfile, index=False)


main()

