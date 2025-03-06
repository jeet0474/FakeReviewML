import pandas as pd
from sklearn.utils import shuffle

# Load the CSV file into a DataFrame
df = pd.read_csv('fake_reviews.csv')
df_label_0 = df[df['label'] == 0]
df_label_1 = df[df['label'] == 1]

#----------------------------------------------------------------------------------------

# Split into test set (2432 rows, 1216 from each label to balance)
test_label_0 = df_label_0.sample(n=1216, random_state=42)
test_label_1 = df_label_1.sample(n=1216, random_state=42)
test = pd.concat([test_label_0, test_label_1])
test = shuffle(test, random_state=42)

#----------------------------------------------------------------------------------------

# Remove test rows from the main DataFrame
df_label_0 = df_label_0.drop(test_label_0.index)
df_label_1 = df_label_1.drop(test_label_1.index)

# Create 4 parts for the remaining set
# Part 1, 2, 3: 11.5k rows each (5750k from label 0, 5750k from label 1)
# For Random Forest, Logistic Regression, and SVM
part1_label_0 = df_label_0[:5750]
part1_label_1 = df_label_1[:5750]
part1 = pd.concat([part1_label_0, part1_label_1])

part2_label_0 = df_label_0[5750:11500]
part2_label_1 = df_label_1[5750:11500]
part2 = pd.concat([part2_label_0, part2_label_1])

part3_label_0 = df_label_0[11500:17250]
part3_label_1 = df_label_1[11500:17250]
part3 = pd.concat([part3_label_0, part3_label_1])

# Part 4: Remaining 3.5k rows (1.75k from label 0 and 1.75k from label 1)
# For XGBoost
part4_label_0 = df_label_0[17250:19000]
part4_label_1 = df_label_1[17250:19000]
part4 = pd.concat([part4_label_0, part4_label_1])

#----------------------------------------------------------------------------------------

# Mixing the parts with other parts for diverse training but still maintaining balance
# For Part 1: Append 1k from each of Part 2 and 3
part1_mixed = pd.concat([part1, part2_label_0[:750], part3_label_1[:750]])
# For Part 2: Append 1k from each of Part 1 and 3
part2_mixed = pd.concat([part2, part1_label_1[:750], part3_label_0[:750]])
# For Part 3: Append 1k from each of Part 1 and 2
part3_mixed = pd.concat([part3, part1_label_0[:750], part2_label_1[:750]])
# For Part 4: Append 1.5k from each of Part 1, 2, and 3
part4_mixed = pd.concat([part4, part1[5000:6500], part2[5000:6500], part3[5000:6500]])

#----------------------------------------------------------------------------------------

# Shuffle each part to ensure randomness
part1_mixed = shuffle(part1_mixed, random_state=42)
part2_mixed = shuffle(part2_mixed, random_state=42)
part3_mixed = shuffle(part3_mixed, random_state=42)
part4_mixed = shuffle(part4_mixed, random_state=42)

# Print the count of 0s and 1s in each part to verify balance
print(part1_mixed['label'].value_counts())
print(part2_mixed['label'].value_counts())
print(part3_mixed['label'].value_counts())
print(part4_mixed['label'].value_counts())

#----------------------------------------------------------------------------------------

# Save the final datasets as CSV
part1_mixed.to_csv('parts/part1.csv', index=False)
part2_mixed.to_csv('parts/part2.csv', index=False)
part3_mixed.to_csv('parts/part3.csv', index=False)
part4_mixed.to_csv('parts/part4.csv', index=False)
test.to_csv('parts/test.csv', index=False)

print("The parts and test set have been saved as CSVs.")