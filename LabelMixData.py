import random
# Paths to the input files



fintech_file_path = 'fintech_data.txt'
non_fintech_file_path = 'non_fintech_data.txt'

# Paths to the output files with labeled data
labeled_fintech_file_path = 'labeled_fintech_data.txt'
labeled_non_fintech_file_path = 'labeled_non_fintech_data.txt'

# Label fintech words as 1
with open(fintech_file_path, 'r') as fin, open(labeled_fintech_file_path, 'w') as fout:
    for word in fin:
        fout.write(f'1 {word.strip()}\n')

# Label non-fintech words as 0
with open(non_fintech_file_path, 'r') as fin, open(labeled_non_fintech_file_path, 'w') as fout:
    for word in fin:
        fout.write(f'0 {word.strip()}\n')

# Combine labeled fintech and non-fintech data into a single file
mixed_labeled_file_path = 'mixed_labeled_data.txt'

# Read labeled fintech data
with open(labeled_fintech_file_path, 'r') as fin:
    labeled_fintech_data = fin.readlines()

# Read labeled non-fintech data
with open(labeled_non_fintech_file_path, 'r') as fin:
    labeled_non_fintech_data = fin.readlines()

# Combine labeled data
mixed_labeled_data = labeled_fintech_data + labeled_non_fintech_data

# Shuffle the combined data
random.shuffle(mixed_labeled_data)

# Write mixed and shuffled labeled data to a file
with open(mixed_labeled_file_path, 'w') as fout:
    fout.writelines(mixed_labeled_data)

print("Mixed and labeled data saved to mixed_labeled_data.txt")
