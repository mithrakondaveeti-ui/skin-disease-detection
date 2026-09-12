"""
Skin Disease Detection Project - Data Preprocessing Pipeline
================================================================
This script handles:
1. Downloading 3 datasets from Kaggle (HAM10000, ISIC 2019, DermNet)
2. Selecting relevant classes from DermNet
3. Combining all datasets into a single labeled DataFrame
4. Train/Validation/Test split (70/15/15, stratified)
5. Class imbalance handling (class weights)
6. Image preprocessing + augmentation setup (ready for model training)

Run this in Google Colab. Total combined dataset: ~40,603 images across 14 classes.
"""

import os
import shutil
import pandas as pd
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ============================================================
# STEP 1: DOWNLOAD DATASETS
# ============================================================
print("Step 1: Downloading datasets from Kaggle...")

ham_path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
isic_path = kagglehub.dataset_download("salviohexia/isic-2019-skin-lesion-images-for-classification")
dermnet_path = kagglehub.dataset_download("shubhamgoel27/dermnet")

print("HAM10000 path:", ham_path)
print("ISIC 2019 path:", isic_path)
print("DermNet path:", dermnet_path)


# ============================================================
# STEP 2: SELECT 5 RELEVANT CLASSES FROM DERMNET
# ============================================================
print("\nStep 2: Selecting 5 common-disease classes from DermNet...")

selected_classes = {
    'Acne and Rosacea Photos': 'acne',
    'Eczema Photos': 'eczema',
    'Psoriasis pictures Lichen Planus and related diseases': 'psoriasis',
    'Scabies Lyme Disease and other Infestations and Bites': 'scabies',
    'Urticaria Hives': 'urticaria'
}

dermnet_selected_path = '/content/dermnet_selected'

for split in ['train', 'test']:
    for original_name, short_name in selected_classes.items():
        src = os.path.join(dermnet_path, split, original_name)
        dst = os.path.join(dermnet_selected_path, split, short_name)
        if os.path.exists(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)

print("DermNet class selection complete.")


# ============================================================
# STEP 3: PROCESS EACH DATASET INTO A COMMON FORMAT
# ============================================================
print("\nStep 3: Processing each dataset into (image_path, label) format...")

# --- HAM10000 ---
df_ham = pd.read_csv(os.path.join(ham_path, 'HAM10000_metadata.csv'))

def find_ham_image_path(image_id):
    for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
        path = os.path.join(ham_path, part, image_id + '.jpg')
        if os.path.exists(path):
            return path
    return None

df_ham['image_path'] = df_ham['image_id'].apply(find_ham_image_path)
df_ham['label'] = df_ham['dx']
df_ham['dataset_source'] = 'HAM10000'

# --- ISIC 2019 (already organized in class folders) ---
isic_classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
isic_data = []
for cls in isic_classes:
    class_folder = os.path.join(isic_path, cls)
    if os.path.exists(class_folder):
        for fname in os.listdir(class_folder):
            isic_data.append({
                'image_id': fname,
                'label': cls.lower(),
                'image_path': os.path.join(class_folder, fname),
                'dataset_source': 'ISIC2019'
            })
df_isic = pd.DataFrame(isic_data)

# --- DermNet (selected classes only) ---
dermnet_classes = ['acne', 'eczema', 'psoriasis', 'scabies', 'urticaria']
dermnet_data = []
for split in ['train', 'test']:
    for cls in dermnet_classes:
        class_folder = os.path.join(dermnet_selected_path, split, cls)
        if os.path.exists(class_folder):
            for fname in os.listdir(class_folder):
                dermnet_data.append({
                    'image_id': fname,
                    'label': cls,
                    'image_path': os.path.join(class_folder, fname),
                    'dataset_source': 'DermNet'
                })
df_dermnet = pd.DataFrame(dermnet_data)

print("HAM10000:", len(df_ham), "rows")
print("ISIC 2019:", len(df_isic), "rows")
print("DermNet:", len(df_dermnet), "rows")


# ============================================================
# STEP 4: COMBINE ALL DATASETS
# ============================================================
print("\nStep 4: Combining all datasets...")

df_combined = pd.concat([
    df_ham[['image_id', 'label', 'image_path', 'dataset_source']],
    df_isic[['image_id', 'label', 'image_path', 'dataset_source']],
    df_dermnet[['image_id', 'label', 'image_path', 'dataset_source']]
], ignore_index=True)

print("Combined dataset total rows:", len(df_combined))
print("\nClass distribution:\n", df_combined['label'].value_counts())


# ============================================================
# STEP 5: TRAIN / VALIDATION / TEST SPLIT (70/15/15, stratified)
# ============================================================
print("\nStep 5: Splitting into train/val/test...")

train_df, temp_df = train_test_split(
    df_combined, test_size=0.3, stratify=df_combined['label'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
)

print("Train set:", len(train_df))
print("Validation set:", len(val_df))
print("Test set:", len(test_df))

# Save splits as CSV for handoff to teammate
train_df.to_csv('/content/train_labels.csv', index=False)
val_df.to_csv('/content/val_labels.csv', index=False)
test_df.to_csv('/content/test_labels.csv', index=False)
print("Saved train_labels.csv, val_labels.csv, test_labels.csv")


# ============================================================
# STEP 6: CLASS WEIGHTS (for handling class imbalance)
# ============================================================
print("\nStep 6: Computing class weights...")

classes = sorted(train_df['label'].unique())
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.array(classes),
    y=train_df['label']
)
class_weights = dict(zip(range(len(classes)), class_weights_array))

print("Classes:", classes)
print("Class weights:", class_weights)


# ============================================================
# STEP 7: IMAGE PREPROCESSING + AUGMENTATION (generators)
# ============================================================
print("\nStep 7: Setting up image generators...")

IMG_SIZE = 224
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.15
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df, x_col='image_path', y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=val_df, x_col='image_path', y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=test_df, x_col='image_path', y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

print("\n" + "="*60)
print("PREPROCESSING COMPLETE - Ready for model training!")
print("="*60)
print("Use: train_generator, val_generator, test_generator, class_weights")
