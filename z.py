import csv

file_path = 'data/advbench/harmful_behaviors.csv'

# Open the CSV file
with open(file_path, mode='r', encoding='utf-8') as file:
    
    reader = csv.reader(file,delimiter=";")

    for row in reader:
        print(row)