import os

def validate_dataset(image_dir, annotation_dir):
    for class_folder in os.listdir(image_dir):
        class_folder_path = os.path.join(image_dir, class_folder)
        if os.path.isdir(class_folder_path):
            for img_file in os.listdir(class_folder_path):
                if img_file.endswith('.jpg'):
                    annotation_file = img_file.replace('.jpg', '.txt')
                    annotation_path = os.path.join(annotation_dir, class_folder, annotation_file)
                    if not os.path.exists(annotation_path):
                        print(f"Annotation file missing for image: {img_file}")
                    else:
                        print(f"Found annotation file for image: {img_file}")

# Replace with actual paths
validate_dataset(r'C:\Users\Amal Anjula\OneDrive - MMU\Subject\MSc project2\trainImages', r'C:\Users\Amal Anjula\OneDrive - MMU\Subject\MSc project2\trainImages')
