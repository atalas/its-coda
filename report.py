import duckdb
import pandas as pd
import matplotlib.pyplot as plt
from   datetime import datetime

#inputFile = "phylum-absolute-sample-study-bodysite.csv"
inputFile = "firmicutes-occurrences.csv"

def main():
	#query = f"SELECT * FROM read_csv_auto('{inputFile}') WHERE "p__Firmicutes" > 0"
	#query = f"SELECT Project, SUM(p__Firmicutes) AS total_occurences FROM read_csv_auto('{inputFile}') GROUP BY Project"
	query = f"SELECT Body_Site, SUM(p__Firmicutes) AS total_occurences FROM read_csv_auto('{inputFile}') GROUP BY Body_Site"

	### REMEMBER TO CHANGE PLOT TITLE ###

	df = runQuery(query)
	createPlot(df, "Body_Site", "total_occurences")
	saveData(df)

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
	plt.title("Firmicutes Occurrences by Body Site")

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
	plt.savefig("firmicutes" + "-" + now + ".png", dpi=300, bbox_inches='tight')


main()

