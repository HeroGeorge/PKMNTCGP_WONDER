import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Rea CSV File typeshii
file_path = 'shuffle_data.csv'  
data = pd.read_csv(file_path)

# Define the number of cards to b e shuffle(5 for PKMNTCGP)
NUM_CARDS = 5

# Create a transition matrix to count occurrences for each iteration (seed)
transition_matrices = {}

# Populate the transition matrices from the data
for index, row in data.iterrows():
    seed = row['seed']
    original = row['original_position'] - 1  # Convert to 0-based index for processing
    shuffled = row['shuffled_position'] - 1  # Convert to 0-based index for processing

    # Initialize the transition matrix for this seed if it doesn't exist
    if seed not in transition_matrices:
        transition_matrices[seed] = np.zeros((NUM_CARDS, NUM_CARDS))
    
    # Update the corresponding transition matrix
    transition_matrices[seed][original][shuffled] += 1

# Normalize matrices to get probabilities
probability_matrices = {}
for seed, matrix in transition_matrices.items():
    total_shuffles = matrix.sum()  # Total counts for this seed
    if total_shuffles > 0:
        probability_matrices[seed] = matrix / total_shuffles  # Normalize

# Create an average probability matrix across all seeds
average_probability_matrix = np.mean(
    [matrix for matrix in probability_matrices.values() if matrix is not None],
    axis=0
)

def predict_final_position(original_position, probability_matrix):
    """Predict the most likely final position of a card."""
    original_index = original_position - 1  # Convert from 1-based to 0-based
    probabilities = probability_matrix[original_index]  # Access the row corresponding to the original position
    most_likely_position = np.argmax(probabilities) + 1  # Convert back to 1-based index
    return most_likely_position, probabilities

# Prediction Model
while True:
    while True:
        try:
            original_position = int(input(f"Enter the original position (1-{NUM_CARDS}): "))
            if 1 <= original_position <= NUM_CARDS:
                break
            else:
                print(f"Please enter a number between 1 and {NUM_CARDS}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    predicted_position, probabilities = predict_final_position(original_position, average_probability_matrix)

    print(f"Original Position: {original_position}")
    print(f"Predicted Final Position: {predicted_position}")
    formatted_probabilities = [f"{p:.3f}" for p in probabilities]  # Format probabilities to 3 decimal places
    print(f"Probabilities of final positions: {formatted_probabilities}")

    # Ask if the user wants to make another prediction
    again = input("Do you want to predict another position? (yes/no): ").strip().lower()
    if again not in ["yes", "y"]:
        print("Goodbye!")
        break

# Visualization of the average probability matrix
plt.figure(figsize=(6, 4))
plt.imshow(average_probability_matrix, cmap="viridis", interpolation="nearest")
plt.colorbar(label="Average Probability")
plt.title("Average Probability Matrix of Card Positions After Shuffle")
plt.xticks(ticks=np.arange(NUM_CARDS), labels=np.arange(1, NUM_CARDS + 1))
plt.yticks(ticks=np.arange(NUM_CARDS), labels=np.arange(1, NUM_CARDS + 1))
plt.xlabel("Final Position")
plt.ylabel("Original Position")
plt.show()
