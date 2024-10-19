import os
import shutil
import pandas as pd
pathb = "/Users/ehushubhamshaw/Desktop/KDD/assignment2/project2_kdd_items/archive/"
train_csv_path = pathb + 'test_m.csv'
image_path_csv_path = pathb + 'image_path.csv'
image_base_directory = pathb + 'jpeg'
output_directory = pathb + 'images_cancer_test'

train_df = pd.read_csv(train_csv_path)
image_path_df = pd.read_csv(image_path_csv_path)
pathology_classes = train_df['pathology'].unique()
for pathology_class in pathology_classes:
    pathology_dir = os.path.join(output_directory, pathology_class)
    os.makedirs(pathology_dir, exist_ok=True)

def segregate_images():
    for _, row in train_df.iterrows():
        image_file = row['image file path']
        pathology = row['pathology'].strip()
        image_info = image_path_df[image_path_df['image_path'] == image_file]
        if not image_info.empty:
            image_name = image_info.iloc[0]['Image_name']
            full_image_path = os.path.join(image_base_directory, image_file, image_name)
            if os.path.exists(full_image_path):
                target_path = os.path.join(output_directory, pathology, image_name)
                shutil.copy(full_image_path, target_path)
                print(f"Copied {full_image_path} to {target_path}")
            else:
                print(f"Image {full_image_path} not found.")
        else:
            print(f"Directory {image_file} not found in image_path.csv.")

segregate_images()
print("Image segregation complete.")