import duckdb
import pandas as pd
import matplotlib.pyplot as plt
from   datetime import datetime

inputFile = "phylum-absolute-sample-study-bodysite.csv"
#inputFile = "firmicutes-occurrences.csv"
#inputFile = "phylum-normalized-sample-study-bodysite.csv"

def main():
	#query = f"SELECT * FROM read_csv_auto('{inputFile}') WHERE "p__Firmicutes" > 0"
	#query = f"SELECT Project, SUM(p__Firmicutes) AS total_occurences FROM read_csv_auto('{inputFile}') GROUP BY Project"
	#query = f"SELECT Body_Site, SUM(p__Firmicutes) AS total_occurences FROM read_csv_auto('{inputFile}') GROUP BY Body_Site"
	query = f"SELECT order_number, AVG(p__Firmicutes) as Average_per_study,  FROM read_csv_auto('{inputFile}') GROUP BY Body_Site"

	### REMEMBER TO CHANGE PLOT TITLE ###

	df = runQuery(query)
	createPlot(df, "Body_Site", "total_occurences")
	saveData(df)


def countPerStudy():
	df = pd.read_csv(inputFile)

	grouped = df.groupby("Project")

	# nunique() counts the number of distinct values in the specified column for each group.
	body_site_counts_series = grouped['Body_Site'].nunique().reset_index(name="Body_Site Count")
	
	saveData(body_site_counts_series)
	createPlot(body_site_counts_series, "Project", "Body_Site Count")	

def ratioPerStudy():
	df = pd.read_csv(inputFile)

	feature_cols = [col for col in df.columns if col not in ['Sample', 'Project', 'Body_Site']]
	grouped = df.groupby("Project")

	# Calculate total Firmicutes per study
	total_firm = grouped["Unassigned"].sum().reset_index(name="Total_Unassigned")

	# Calculate total features count per study
	total_all_features = grouped[feature_cols].sum().sum(axis=1).reset_index(level=0, drop=True).reset_index(drop=True)
	total_all_features.columns = ["Total_Features"]

	result_df = total_firm.copy()
	result_df["Total_Features_Combined"] = total_all_features
	result_df["Percentage_Of_Unassigned"] = (result_df["Total_Unassigned"] / result_df["Total_Features_Combined"]) * 100
	saveData(result_df)
	createPlot(result_df, "Project", "Percentage_Of_Unassigned")


def saveData(df):
	now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
	df.to_csv("output" + now + ".csv", index=False)


def runQuery(query):
	con = duckdb.connect(database=":memory:")

	# Execute and fetch the result into a pandas DataFrame
	return con.execute(query).fetchdf()


def createPlot(result, x, y):
	# Create a figure with a wide aspect ratio
	fig, ax = plt.subplots(figsize=(20, 8))

	plt.xlabel(x)
	plt.ylabel(y)
	plt.title("Percent Of Unassigned by Study")

	# Plot bars with explicit width (e.g., 0.6 for wider bars)
	bars = ax.bar(
		result[x],
		result[y],
		width=0.6,  # Adjust this value to make bars wider
		align='center'
	)

	plt.tight_layout()
	plt.xticks(rotation=80, ha='right')
	#plt.show()

	now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
	plt.savefig("barplot" + "-" + now + ".png", dpi=300, bbox_inches='tight')

#main()
ratioPerStudy()
#countPerStudy()

