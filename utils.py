import glob
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import datasets as dts

from config import CFG

def get_sample_patient_id(image_paths):
    return [(_.split('/')[-2:][0]) for _ in image_paths]

def get_sample_number(image_paths):
    sample_numbers = []
    is_mask = []

    for path in image_paths:
        path_list = path.split('/')[-2:][1].split('_')

        if 'mask.tif' in path_list:
            sample_numbers.append(int(path_list[-2]))
            is_mask.append(1)
        else:
            sample_numbers.append(int(path_list[-1].replace('.tif', '')))
            is_mask.append(0)

    return sample_numbers, is_mask

def build_df(image_paths):
    sample_numbers, mask_label = get_sample_number(image_paths)
    df = pd.DataFrame({
        'id'        : sample_numbers,
        'patient'   : get_sample_patient_id(image_paths),
        'image_path': image_paths,
        'is_mask'   : mask_label
    })
    return df

def _load(image_path, as_tensor=True):
    image = Image.open(image_path)
    return np.array(image).astype(np.float32) / 255.

def generate_label(mask_path, load_fn):
    mask = load_fn(mask_path)
    if mask.max() > 0:
        return 1 # Brain Tumor Present
    return 0 # Normal

def transform(data):
    with open(data['image_path'], 'rb') as f:
        image = Image.open(f).convert('RGB')
    data['image'] = image

    with open(data['mask_path'], 'rb') as f:
        mask = Image.open(f).convert('L') # to grayscale
    data['mask'] = mask

    return data

def prepare_data():
    dataset_images = glob.glob(f"{CFG.TRAIN_PATH}**/*.tif")
    dataset_df = (
        build_df(dataset_images)
        .sort_values(by=['id', 'patient', 'image_path'])
        .reset_index(drop=True)
    )

    grouped_df = dataset_df.groupby(by='is_mask')
    images_df, mask_df = (
        grouped_df.get_group(0).drop('is_mask', axis=1).reset_index(drop=True),
        grouped_df.get_group(1).drop('is_mask', axis=1).reset_index(drop=True)
    )

    mask_df = mask_df.rename({'image_path': 'mask_path'}, axis=1)

    ds = images_df.merge(
        mask_df,
        on=['id', 'patient'],
        how='left'
    )

    ds['diagnosis'] = [generate_label(_, _load) for _ in tqdm(ds['mask_path'])]
    ds = ds[ds['diagnosis']==1]
    ds = ds.head(1360)

    image_train, image_test, mask_train, mask_test = train_test_split(
        ds['image_path'], ds['mask_path'], test_size = 0.10)

    train_df = pd.concat([image_train, mask_train], axis=1).reset_index(drop=True)
    train_dataset = dts.Dataset.from_pandas(train_df)

    test_df = pd.concat([image_test, mask_test], axis=1).reset_index(drop=True)
    test_dataset = dts.Dataset.from_pandas(test_df)

    train_dataset = train_dataset.map(transform, remove_columns=['image_path','mask_path'])
    test_dataset = test_dataset.map(transform, remove_columns=['image_path','mask_path'])

    return train_dataset, test_dataset
