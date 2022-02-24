import csv
import os


if not os.path.exists("../data"):
    raise Exception("The specified path {} does not exist.".format("../data"))
    # Initialize the data.
data = list(range(1775))
# Iterate over the row to fill in the data.
with open("../data/question_meta.csv", "r") as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        try:
            data[int(row[0])] = row[1][1:-1].split(",")
        except ValueError:
            # Pass first row.
            pass
        except IndexError:
            # is_correct might not be available.
            pass

print(data)
i = int(data[0][1].strip())
print(type(i))