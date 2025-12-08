import numpy as np
from PIL import Image
import os
import cmath
import math

url_path = "imagenes/img.bmp"

sizeBytes = os.path.getsize(url_path)

print(f"Tamaño de la imagen original: {sizeBytes / 1024:.2f} KB")
img = Image.open(url_path)

image = np.array(img.convert('L'))  # convertimos la imagen a escala de grises y la almacenamos en un array

def dft(x):
    N = len(x)
    resultado = []
    
    for k in range(N):
        suma = 0
        for n in range(N):
            suma += x[n] * cmath.exp(-2j * math.pi * k * n / N)
        resultado.append(suma)
    
    return resultado


def idft(x):
    N = len(x)
    resultado = []
    
    for n in range(N):
        suma = 0
        for k in range(N):
            suma += x[k] * cmath.exp(2j * math.pi * k * n / N)
        resultado.append(suma / N)
    
    return resultado


def dft2d(imagen):
    M, N = imagen.shape
    print(f"Aplicando DFT 2D a imagen de {M}x{N}...")
    
    print("  DFT en filas...")
    filas_dft = []
    for i, fila in enumerate(imagen):
        print(f"    Procesando fila {i+1}/{M}")
        filas_dft.append(dft(fila))
    filas_dft = np.array(filas_dft)
    
    print("  DFT en columnas...")
    columnas_dft = []
    for j, columna in enumerate(filas_dft.T):
        print(f"    Procesando columna {j+1}/{N}")
        columnas_dft.append(dft(columna))
    columnas_dft = np.array(columnas_dft).T
    
    return columnas_dft


def idft2d(dft_image, original_shape):
    print("Aplicando IDFT 2D...")
    
    print("  IDFT en columnas...")
    columnas_idft = []
    for j, columnas in enumerate(dft_image.T):
        print(f"    Procesando columna {j+1}/{dft_image.shape[1]}")
        columnas_idft.append(idft(columnas))
    columnas_idft = np.array(columnas_idft).T
    
    print("  IDFT en filas...")
    filas_idft = []
    for i, fila in enumerate(columnas_idft):
        print(f"    Procesando fila {i+1}/{dft_image.shape[0]}")
        filas_idft.append(idft(fila))
    filas_idft = np.array(filas_idft)
    
    # Recortar al tamaño original
    M, N = original_shape
    return filas_idft[:M, :N].real


porcentaje_conservar = 1

print(f"\nComprimiendo imagen con {porcentaje_conservar}% de coeficientes...")

# Aplicar DFT 2D
dft_image = dft2d(image)

print(image.shape[0] * image.shape[1], "pixeles en total")
print(len(dft_image.flatten()), "coeficientes en total")

# Calcular magnitudes
magnitudes = np.abs(dft_image)

# Determinar umbral para conservar solo el porcentaje deseado de coeficientes
umbral = np.percentile(magnitudes, 100 - porcentaje_conservar)

# Crear máscara para conservar solo los coeficientes más importantes
mascara = magnitudes >= umbral
dft_comprimida = dft_image * mascara

coef_conservados = np.sum(mascara)
coef_totales = mascara.size
print(f"Coeficientes conservados: {coef_conservados} de {coef_totales} ({100*coef_conservados/coef_totales:.2f}%)")

# Aplicar DFT inversa
imagen_reconstruida = idft2d(dft_comprimida, image.shape)

# Asegurar que los valores estén en el rango correcto
imagen_reconstruida = np.clip(imagen_reconstruida, 0, 255).astype(np.uint8)

# Calcular tamaño aproximado (solo coeficientes no nulos)
coef_no_nulos = np.count_nonzero(dft_comprimida)
# Cada coeficiente complejo = 2 floats de 64 bits = 16 bytes
tamaño_estimado = coef_no_nulos * 16
print(f"Tamaño estimado de la imagen comprimida: {tamaño_estimado / 1024:.2f} KB")

Image.fromarray(imagen_reconstruida).save('imagenes/img_dft.bmp')
print(f"\nResultado guardado en: imagenes/img_dft.bmp")
