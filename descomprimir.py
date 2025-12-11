import numpy as np
from PIL import Image
import os
import cmath
import math
import struct

def ifft(x):
    N = len(x)
    
    # Conjugar cada valor de la entrada
    x_conj = []
    for valor in x:
        valor_conjugado = complex(valor).conjugate()
        x_conj.append(valor_conjugado)
    
    # Aplicar FFT a la secuencia conjugada
    resultado = fft(x_conj)
    
    # Conjugar la salida y dividir por N
    resultado_final = []
    for valor in resultado:
        valor_conjugado_normalizado = complex(valor).conjugate() / N
        resultado_final.append(valor_conjugado_normalizado)
    
    return resultado_final

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

def ifft2d(imagen_fft, dimensiones_originales):
    transpuesta = imagen_fft.T
    
    columnas_ifft = []
    for col in transpuesta:
        transformada_inversa = ifft(col)
        columnas_ifft.append(transformada_inversa)
    
    columnas_ifft = np.array(columnas_ifft).T
    
    # IFFT en las filas
    filas_ifft = []
    for row in columnas_ifft:
        transformada_inversa = ifft(row)
        filas_ifft.append(transformada_inversa)
    
    filas_ifft = np.array(filas_ifft)
    
    # Recortar al tamaño original
    M, N = dimensiones_originales
    return filas_ifft[:M, :N].real

archivo_comprimido = 'imagenes/img_comprimida.bin'

print(f"Leyendo archivo: {archivo_comprimido}")

tamaño_archivo = os.path.getsize(archivo_comprimido)
print(f"Tamaño del archivo comprimido: {tamaño_archivo / 1024:.2f} KB")

# Leer el archivo binario
with open(archivo_comprimido, 'rb') as f:
    # Leer dimensiones originales (2 bytes cada una)
    alto_original = struct.unpack('H', f.read(2))[0]
    ancho_original = struct.unpack('H', f.read(2))[0]
    
    # Leer dimensiones de la matriz FFT (2 bytes cada una)
    alto_fft = struct.unpack('H', f.read(2))[0]
    ancho_fft = struct.unpack('H', f.read(2))[0]
    
    # Leer número de coeficientes (4 bytes)
    num_coeficientes = struct.unpack('I', f.read(4))[0]
    
    print(f"Dimensiones originales: {alto_original}x{ancho_original}")
    print(f"Dimensiones FFT: {alto_fft}x{ancho_fft}")
    print(f"Coeficientes almacenados: {num_coeficientes}")
    
    # Reconstruir la matriz FFT
    fft_reconstruida = np.zeros((alto_fft, ancho_fft), dtype=complex)
    
    # Leer los coeficientes no nulos
    for _ in range(num_coeficientes):
        fila = struct.unpack('H', f.read(2))[0]
        columna = struct.unpack('H', f.read(2))[0]
        parte_real = struct.unpack('f', f.read(4))[0]
        parte_imag = struct.unpack('f', f.read(4))[0]
        
        fft_reconstruida[fila, columna] = complex(parte_real, parte_imag)

print(f"\nAplicando FFT inversa...")

# Aplicar FFT inversa
imagen_reconstruida = ifft2d(fft_reconstruida, (alto_original, ancho_original))

# Asegurar que los valores estén en el rango correcto
imagen_reconstruida = np.clip(imagen_reconstruida, 0, 255).astype(np.uint8)

# Guardar la imagen reconstruida
ruta_imagen_descomprimida = 'imagenes/img_descomprimida.png'
Image.fromarray(imagen_reconstruida).save(ruta_imagen_descomprimida)

print(f"\nImagen descomprimida guardada en: {ruta_imagen_descomprimida}")
