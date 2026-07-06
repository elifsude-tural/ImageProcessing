import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import albumentations as A


class DigitDataGenerator(Sequence):
    """
    keras.utils.Sequence'dan extend eden custom data generator.
    data/train/<digit>/*.png  veya  data/val/<digit>/*.png yapısını okur.
    augment=True verilirse (sadece train icin) rastgele donme/kaydirma/zoom uygular.
    """

    def __init__(self, data_dir, img_size=(28, 28), batch_size=128,
                 augment=False, shuffle=True):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle

        
        self.samples = [] 
        class_names = sorted(os.listdir(data_dir))  
        for class_name in class_names:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            label = int(class_name)
            for fname in os.listdir(class_dir):
                self.samples.append((os.path.join(class_dir, fname), label))

        
        if self.augment:
            self.aug_pipeline = A.Compose([
                A.Rotate(limit=15, p=0.5),                             
                A.Affine(translate_percent=0.1, scale=(0.9, 1.1), p=0.5),  
                A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),            
                A.RandomBrightnessContrast(p=0.2),                    
            ])

        self.on_epoch_end()

    def __len__(self):
       
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, idx):
        batch_samples = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        X_batch = np.zeros((len(batch_samples), *self.img_size, 1), dtype="float32")
        y_batch = np.zeros((len(batch_samples),), dtype="int32")

        for i, sample_idx in enumerate(batch_samples):
            filepath, label = self.samples[sample_idx]
            img = load_img(filepath, color_mode="grayscale", target_size=self.img_size)
            
            img_array = img_to_array(img) 
            

        if self.augment:
            img_array = self.aug_pipeline(image=img_array)["image"]

        X_batch[i] = img_array / 255.0  
        y_batch[i] = label
            

        return X_batch, y_batch

    def on_epoch_end(self):
    
        self.indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indices)
