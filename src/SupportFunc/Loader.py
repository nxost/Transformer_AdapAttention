import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class MultiLabelImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (str): Ruta del archivo CSV con nombres de archivo y etiquetas.
            img_dir (str): Ruta a la carpeta de imágenes.
            transform (callable, optional): Transformaciones a aplicar a las imágenes.
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Obtener el ID del archivo desde el CSV
        img_id = self.data.iloc[idx, 0]  # Primera columna (ID)
        extensions = ["", ".jpg", ".png", ".tif"]
        img_path = None

        # Buscar la imagen con diferentes extensiones
        for ext in extensions:
            temp_path = os.path.join(self.img_dir, str(img_id) + ext)
            if os.path.exists(temp_path):
                img_path = temp_path
                break

        if img_path is None:
            raise FileNotFoundError(
                f"No se encontró la imagen para el ID '{img_id}' en {self.img_dir} con extensiones {extensions}."
            )

        # Cargar la imagen
        image = Image.open(img_path).convert("RGB")  # Convertir a RGB
        if self.transform:
            image = self.transform(image)

        # Leer etiquetas multi-label
        labels = torch.tensor(self.data.iloc[idx, 1:].values.astype(float))
        return image, labels
