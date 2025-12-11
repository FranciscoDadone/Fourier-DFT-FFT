import numpy as np
from PIL import Image
import os
import cmath
import math

url_path = "imagenes/img_chica.png"

sizeBytes = os.path.getsize(url_path)

print(f"Tamaño de la imagen original: {sizeBytes / 1024:.2f} KB")

img = Image.open(url_path)

# convertimos la imagen a escala de grises y la almacenamos en un array
image = np.array(img.convert('L'))  

def fft(x):
    N = len(x)
    if N <= 1:
        return x
    elementos_par = x[0::2]
    elementos_impar = x[1::2]
    fft_par = fft(elementos_par)
    fft_impar  = fft(elementos_impar)
    
    # Matriz nueva (de 0s) para almacenar el resultado
    combinado = [0] * N
    
    for k in range(N // 2):
        # Factor de rotación
        W = cmath.exp(-2j * math.pi * k / N)
        
        t = W * fft_impar[k]
        
        combinado[k]          = fft_par[k] + t
        combinado[k + N // 2] = fft_par[k] - t
    return combinado

def ifft(x):
    N = len(x)
    
    # Conjugar cada valor de la entrada
    x_conj = []
    for valor in x:
        # Cambio de signo de la parte imaginaria
        valor_conjugado = complex(valor).conjugate()
        x_conj.append(valor_conjugado)
    
    # Aplicar FFT a la secuencia conjugada
    resultado = fft(x_conj)
    
    # Conjugar la salida y dividir por N para obtener la transformada inversa
    resultado_final = []
    for valor in resultado:
        valor_conjugado_normalizado = complex(valor).conjugate() / N
        resultado_final.append(valor_conjugado_normalizado)
    
    return resultado_final

def fft2d(image):
    M, N = image.shape # dimensiones de la imagen
    
    # Asegurar que las dimensiones sean potencias de 2 ya que lo requerimos a esto para hacer la FFT
    nueva_M = 2 ** int(np.ceil(np.log2(M)))
    nueva_N = 2 ** int(np.ceil(np.log2(N)))
    
    matriz = np.zeros((nueva_M, nueva_N), dtype=complex) # Hago una matriz de la imagen con ceros
    matriz[:M, :N] = image # pongo la imagen en la nueva matriz potencia de 2    
    
    filas_fft = []
    for fila in matriz:
        transformada = fft(fila)   
        filas_fft.append(transformada)

    filas_fft = np.array(filas_fft)

    # Transponer, aplicar FFT, transponer de vuelta
    transpuesta = filas_fft.T

    # Aplicar FFT a cada fila (que representa columnas originales)
    columnas_fft = []
    for col in transpuesta:
        transformada = fft(col)
        columnas_fft.append(transformada)

    # Volver a forma original
    fft_cols = np.array(columnas_fft).T
    
    return fft_cols

def ifft2d(imagen_fft, dimensiones_originales):
    transpuesta = imagen_fft.T
    
    columnas_ifft = []
    for col in transpuesta:
        transformada_inversa = ifft(col)
        columnas_ifft.append(transformada_inversa)
    
    columnas_ifft = np.array(columnas_ifft).T
    
    # IFFT en las filas
    filas_ifft = []
    for fila in columnas_ifft:
        transformada_inversa = ifft(fila)
        filas_ifft.append(transformada_inversa)
    
    filas_ifft = np.array(filas_ifft)
    
    # Recortar al tamaño original
    M, N = dimensiones_originales
    return filas_ifft[:M, :N].real

porcentaje_conservar = 1

# Mostrar imagen comprimida

print("\n=== COMPRIMIENDO IMAGEN ===")
print("Haciendo la FFT 2D de la imagen...")
fft_image = fft2d(image)
# fft_image = np.abs(fft_image).astype(complex) # demo fase 0
print(image.shape[0] * image.shape[1], "pixeles en total")
print(len(fft_image.flatten()), "coeficientes en total")

# Calcula el módulo complejo de cada coeficiente
magnitudes = np.abs(fft_image)

# Determinar umbral para conservar solo el porcentaje deseado de coeficientes
umbral = np.percentile(magnitudes, 100 - porcentaje_conservar)

# Matriz de 0s y 1s para conservar solo los coeficientes más importantes
mascara = magnitudes >= umbral

# Aplicar la máscara
fft_comprimida = fft_image * mascara

coef_conservados = np.sum(mascara)
coef_totales = mascara.size
print(f"Coeficientes conservados: {coef_conservados} de {coef_totales} ({100*coef_conservados/coef_totales:.2f}%)")


# Aplicar FFT inversa
print("Haciendo la IFFT 2D de la imagen comprimida...")
imagen_reconstruida = ifft2d(fft_comprimida, image.shape)

# Asegurar que los valores estén en el rango correcto (8 bits)
imagen_reconstruida = np.clip(imagen_reconstruida, 0, 255).astype(np.uint8)

coef_no_nulos = np.count_nonzero(fft_comprimida)

# Cada coeficiente complejo = 2 floats de 64 bits = 16 bytes
# El tamaño estimado es antes de hacer la IFFT
tamaño_estimado = 10 + (coef_no_nulos * 12)
print(f"Tamaño estimado de la imagen comprimida: {tamaño_estimado / 1024:.2f} KB")

# Guardar la imagen comprimida en un archivo
ruta_imagen_comprimida = 'imagenes/fft_img_chica.png'
Image.fromarray(imagen_reconstruida).save(ruta_imagen_comprimida)
print(f"\nImagen comprimida guardada en: {ruta_imagen_comprimida}")

# Guardar imagen de la máscara
# mascara_img = (mascara * 255).astype(np.uint8)
# ruta_mascara = 'imagenes/fft_mascara.png'
# Image.fromarray(mascara_img).save(ruta_mascara)
# print(f"Imagen de la máscara guardada en: {ruta_mascara}")



# fft_image_img = (fft_image * 255).astype(np.uint8)
# ruta_fft_image = 'imagenes/fft_image.png'
# Image.fromarray(fft_image_img).save(ruta_fft_image)
# print(f"Imagen de la FFT guardada en: {ruta_fft_image}")


# fft_image_comprimida_img = (fft_comprimida * 255).astype(np.uint8)
# ruta_fft_image_comprimida = 'imagenes/fft_image_comprimida.png'
# Image.fromarray(fft_image_comprimida_img).save(ruta_fft_image_comprimida)
# print(f"Imagen de la FFT comprimida guardada en: {ruta_fft_image_comprimida}")