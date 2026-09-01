import duckdb
import pandas as pd

inputFile = "phylum-absolute-sample-study-bodysite.csv"

con = duckdb.connect(database=":memory:")

query = f"""
	SELECT *
	FROM read_csv_auto('{inputFile}')
	WHERE "d__Bacteria;p__Firmicutes" > 0
"""

# Execute and fetch the result into a pandas DataFrame
df = con.execute(query).fetchdf()

df.to_csv('output.csv', index=False)



