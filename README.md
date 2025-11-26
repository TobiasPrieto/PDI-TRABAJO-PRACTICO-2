# Trabajo Práctico 2 - Procesamiento Digital de Imágenes (PDI)

## Integrantes
- Esteva Matias (E-1253/1)
- Prieto Tobias (P-5260/4)

Este repositorio contiene la solución al Trabajo Práctico 2 de la materia Procesamiento Digital de Imágenes. El proyecto se divide en dos ejercicios principales que utilizan técnicas de visión por computadora para la detección, clasificación y conteo de objetos.

## Contenido del Repositorio

### Scripts Principales

*   **`tp2_ejercicio1.py`**:
    *   **Objetivo**: Detección y clasificación de monedas y dados en una imagen.
    *   **Funcionalidad**:
        *   **Monedas**: Utiliza la Transformada de Hough para detectar círculos y análisis de color en el espacio HSV para clasificar las monedas en tres categorías: 1 Peso (Bimetal), 50 Centavos (Marrón) y 10 Centavos (Plateada).
        *   **Dados**: Emplea operaciones morfológicas para segmentar los cuerpos de los dados y conteo de componentes conexas para determinar el valor (número de puntos) de cada dado.
    *   **Entrada**: `monedas.jpg`.
    *   **Salida**: Muestra una ventana con dos subplots: uno con las monedas clasificadas y otro con los dados detectados y su suma total.

*   **`tp2_ejercicio2.py`**:
    *   **Objetivo**: Detección y segmentación de patentes vehiculares.
    *   **Funcionalidad**: Procesa un conjunto de imágenes (`img01.png` a `img12.png`) para localizar la patente basándose en características morfológicas y geométricas (relación de aspecto). Posteriormente, segmenta y resalta los caracteres individuales dentro de la patente detectada.
    *   **Entrada**: Imágenes `img01.png` a `img12.png`.
    *   **Salida**: Muestra secuencialmente las imágenes procesadas con la patente y sus caracteres resaltados.

### Imágenes

*   `monedas.jpg`: Imagen utilizada para el Ejercicio 1.
*   `img01.png` - `img12.png`: Conjunto de imágenes de vehículos utilizadas para el Ejercicio 2.

## Requisitos

Para ejecutar los scripts, es necesario tener instalado Python y las siguientes librerías:

*   `opencv-python` (cv2)
*   `numpy`
*   `matplotlib`

Puedes instalarlas utilizando pip:

```bash
pip install opencv-python numpy matplotlib
```

## Instrucciones de Ejecución

### Ejercicio 1: Monedas y Dados

Ejecuta el siguiente comando en la terminal:

```bash
python tp2_ejercicio1.py
```

Se abrirá una ventana de Matplotlib mostrando los resultados de la clasificación de monedas y el conteo de dados. Además, se imprimirá en la consola el conteo detallado de monedas y dados.

### Ejercicio 2: Patentes

Ejecuta el siguiente comando en la terminal:

```bash
python tp2_ejercicio2.py
```

El script procesará las imágenes una por una. Se mostrarán ventanas emergentes con los resultados de la detección para cada imagen. Debes cerrar la ventana actual para que el script proceda a mostrar la siguiente imagen.
