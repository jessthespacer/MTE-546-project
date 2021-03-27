import csv
import numpy as np

with open('MVPcase.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    with open('MVPwithnoise.csv', mode='w', newline='') as write_file:
        file_writer = csv.writer(write_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                file_writer.writerow(row)
            else:
                row = [float(i) for i in row]
                row[3] += np.random.normal(0, 1)
                row[4] += np.random.normal(0, 1)

                file_writer.writerow(row)

                line_count += 1
