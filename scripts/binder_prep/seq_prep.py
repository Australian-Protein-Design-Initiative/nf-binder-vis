import csv

input_file = 'twist_2stop.csv'
output_file = 'output_pad.csv'
dna_padding_file = 'DNA_padding.txt'

# Read the DNA padding sequence from the file
with open(dna_padding_file, 'r') as dna_file:
    dna_padding_sequence = dna_file.read().strip()

# Function to process each row based on conditions
def process_row(row):
    if row[1] == 'N':
        row[3] += 'TGA'
        if int(row[2]) < 297:
            padding_length_n = 297 - int(row[2])
            row[3] += dna_padding_sequence[:padding_length_n]
    elif row[1] == 'C' and int(row[2]) < 300:
        row[3] += 'GGCTCCCACCACCACCACCACCACTGA'
        if int(row[2]) < 273:
            padding_length_c = 273 - int(row[2])
            row[3] += dna_padding_sequence[:padding_length_c]
    return row

# Read the input CSV file and process each row
with open(input_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = [row for row in reader]

# Apply the processing function to each row
processed_data = [process_row(row) for row in data]

# Write the processed data to the output CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(processed_data)

print(f"Processing complete. Results saved to {output_file}")
