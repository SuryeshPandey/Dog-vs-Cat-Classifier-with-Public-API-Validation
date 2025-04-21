import pandas as pd

# Load the ground truth CSV file
ground_truth_path = "F:/suryesh project/vision_pipeline/ground_truth.csv"
ground_df = pd.read_csv(ground_truth_path)

# Mapping of breeds to 'dog' or 'cat'
breed_to_class = {
    'Abyssinian': 'cat',
    'Bengal': 'cat',
    'Birman': 'cat',
    'Bombay': 'cat',
    'British Shorthair': 'cat',
    'Egyptian Mau': 'cat',
    'Persian': 'cat',
    'Ragdoll': 'cat',
    'Russian Blue': 'cat',
    'Siamese': 'cat',
    'Sphynx': 'cat',
    
    # Dog breeds
    'american bulldog': 'dog',
    'american pit bull terrier': 'dog',
    'basset hound': 'dog',
    'beagle': 'dog',
    'boxer': 'dog',
    'chihuahua': 'dog',
    'english cocker spaniel': 'dog',
    'english setter': 'dog',
    'german shorthaired': 'dog',
    'great pyrenees': 'dog',
    'havanese': 'dog',
    'japanese chin': 'dog',
    'keeshond': 'dog',
    'leonberger': 'dog',
    'miniature pinscher': 'dog',
    'newfoundland': 'dog',
    'pomeranian': 'dog',
    'pug': 'dog',
    'saint bernard': 'dog',
    'samoyed': 'dog',
    'scottish terrier': 'dog',
    'shiba inu': 'dog',
    'staffordshire bull terrier': 'dog',
    'wheaten terrier': 'dog',
    'yorkshire terrier': 'dog'
}

# Create a new column 'class_name' based on the 'true_class' column
ground_df['class_name'] = ground_df['true_class'].map(breed_to_class)

# Save the updated DataFrame back to CSV
ground_df.to_csv("F:/suryesh project/vision_pipeline/ground_truth_with_class.csv", index=False)

# Print the first few rows to confirm
print(ground_df.head())
