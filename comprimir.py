import numpy as np
from PIL import Image
import os
import cmath
import math
import struct

url_path = "imagenes/img.png"

tamaño = os.path.getsize(url_path)
print(f"Tamaño de la imagen original: {tamaño / 1024:.2f} KB")

img = Image.open(url_path)

# Convertir la imagen a escala de grises
image = np.array(img.convert('L'))  

def fft(x):
    N = len(x)
    if N <= 1:
        return x
    elementos_par = x[0::2]
    elementos_impar = x[1::2]
    fft_par = fft(elementos_par)
    fft_impar  = fft(elementos_impar)
    
    combinado = [0] * N
    
    for k in range(N // 2):
        W = cmath.exp(-2j * math.pi * k / N)
        t = W * fft_impar[k]
        combinado[k]          = fft_par[k] + t
        combinado[k + N // 2] = fft_par[k] - t
    return combinado

def fft2d(image):
    M, N = image.shape
    
    # Asegurar que las dimensiones sean potencias de 2
    nueva_M = 2 ** int(np.ceil(np.log2(M)))
    nueva_N = 2 ** int(np.ceil(np.log2(N)))
    
    matriz = np.zeros((nueva_M, nueva_N), dtype=complex)
    matriz[:M, :N] = image
    
    filas_fft = []
    for row in matriz:
        transformed_row = fft(row)
        filas_fft.append(transformed_row)

    filas_fft = np.array(filas_fft)

    # Transponer, aplicar FFT, transponer de vuelta
    transpuesta = filas_fft.T

    columnas_fft = []
    for col in transpuesta:
        transformada = fft(col)
        columnas_fft.append(transformada)

    fft_cols = np.array(columnas_fft).T
    
    return fft_cols

# Parámetro de compresión (porcentaje de coeficientes a conservar)
porcentaje_conservar = 1 

print(f"\n=== COMPRIMIENDO IMAGEN ===")
print(f"Dimensiones originales: {image.shape}")
print(f"Pixeles totales: {image.shape[0] * image.shape[1]}")

# Aplicar FFT 2D
fft_image = fft2d(image)
print(f"Coeficientes FFT totales: {len(fft_image.flatten())}")

# Calcular magnitudes
magnitudes = np.abs(fft_image)

# Determinar umbral para conservar solo el porcentaje deseado
umbral = np.percentile(magnitudes, 100 - porcentaje_conservar)

# Crear máscara
mascara = magnitudes >= umbral

# Aplicar la máscara
fft_comprimida = fft_image * mascara

coef_conservados = np.sum(mascara)
coef_totales = mascara.size
print(f"Coeficientes conservados: {coef_conservados} de {coef_totales} ({100*coef_conservados/coef_totales:.2f}%)")

# Obtener solo los coeficientes no nulos
indices_no_nulos = np.nonzero(mascara)
valores_no_nulos = fft_comprimida[indices_no_nulos]

# Guardar en archivo binario
archivo_comprimido = 'imagenes/img_comprimida.bin'

with open(archivo_comprimido, 'wb') as f:
    # Guardar dimensiones originales de la imagen
    f.write(struct.pack('H', image.shape[0]))  # Alto original (2 bytes)
    f.write(struct.pack('H', image.shape[1]))  # Ancho original (2 bytes)
    
    # Guardar dimensiones de la matriz FFT (potencia de 2)
    f.write(struct.pack('H', fft_image.shape[0]))  # Alto FFT (2 bytes)
    f.write(struct.pack('H', fft_image.shape[1]))  # Ancho FFT (2 bytes)
    
    # Guardar número de coeficientes no nulos
    f.write(struct.pack('I', len(valores_no_nulos)))  # 4 bytes
    
    for i in range(len(valores_no_nulos)):
        fila = indices_no_nulos[0][i]
        columna = indices_no_nulos[1][i]
        valor = valores_no_nulos[i]
        
        # Guardar índices (2 bytes cada uno)
        f.write(struct.pack('H', fila))
        f.write(struct.pack('H', columna))
        
        # Guardar parte real e imaginaria (4 bytes cada uno)
        f.write(struct.pack('f', valor.real))
        f.write(struct.pack('f', valor.imag))

tamaño_comprimido = os.path.getsize(archivo_comprimido)
print(f"\nTamaño del archivo comprimido: {tamaño_comprimido / 1024:.2f} KB")
print(f"Reducción de tamaño: {100 * (1 - tamaño_comprimido / tamaño):.2f}%")
print(f"\nArchivo guardado en: {archivo_comprimido}")
