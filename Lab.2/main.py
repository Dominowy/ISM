import os
import pydicom
import matplotlib.pyplot as plt
from pydicom.pixel_data_handlers.util import apply_modality_lut

def show_dicom_image(file_path):
    try:
        dataset = pydicom.dcmread(file_path)

        if 'PixelData' in dataset:
            if 'ModalityLUTSequence' in dataset:
                data = apply_modality_lut(dataset.pixel_array, dataset)
            else:
                data = dataset.pixel_array
            plt.imshow(data, cmap=plt.cm.gray)
            plt.title(f"Obraz DICOM: {file_path}")
            plt.axis('off')
            plt.show()
        else:
            print(f"Plik DICOM {file_path} nie zawiera danych pikseli.")
    except Exception as e:
        print(f"Nie udało się odczytać pliku DICOM {file_path}: {e}")

def browse_dicom_directory(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.dcm'):
                file_path = os.path.join(root, file)
                show_dicom_image(file_path)

dicom_directory_path = './Wariant2/Wariant2/series-00000'
browse_dicom_directory(dicom_directory_path)