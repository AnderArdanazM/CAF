from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import numpy as np
import os

# Ruta al dataset original y a dónde guardar los nuevos archivos .bin
DATASET_PATH = "C:/Users/usuario/Desktop/CAF/v1.0-mini_canbus-001/v1.0-mini"
SAVE_PATH = "C:/Users/usuario/Desktop/CAF/lidar_filtrado_180"
os.makedirs(SAVE_PATH, exist_ok=True)

# Cargar el dataset
nusc = NuScenes(version='v1.0-mini', dataroot=DATASET_PATH, verbose=True)

# Procesar todos los samples
for i, sample in enumerate(nusc.sample):
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    lidar_path = os.path.join(nusc.dataroot, lidar_data['filename'])

    # Cargar nube de puntos LIDAR
    pc = LidarPointCloud.from_file(lidar_path)
    points = pc.points  # (4, N): x, y, z, reflectivity

    # Calcular ángulo horizontal
    x = points[0, :]
    y = points[1, :]
    angles = np.arctan2(y, x)

    # Crear máscara para campo frontal (-90° a +90°)
    mask = (angles > -np.pi / 2) & (angles < np.pi / 2)
    filtered_points = points[:, mask]

    # Guardar puntos filtrados en nuevo archivo .bin
    save_filename = f"lidar_{i:04d}.bin"
    save_path = os.path.join(SAVE_PATH, save_filename)
    filtered_points.T.astype(np.float32).tofile(save_path)

    print(f"[{i+1}/{len(nusc.sample)}] Guardado: {save_filename} → {filtered_points.shape[1]} puntos")
