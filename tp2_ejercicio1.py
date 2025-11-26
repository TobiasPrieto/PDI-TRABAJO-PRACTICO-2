import cv2
import numpy as np
import matplotlib.pyplot as plt

ruta = 'monedas.jpg'
img_original = cv2.imread(ruta)

img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
img_gris = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)

img_blur = cv2.medianBlur(img_gris, 7)



# ==========================================
# PARTE 1: DETECCIÓN Y CLASIFICACIÓN DE MONEDAS
# ==========================================

circulos = cv2.HoughCircles(
    img_blur,
    cv2.HOUGH_GRADIENT,
    dp=1, minDist=100, param1=150, param2=40, minRadius=90, maxRadius=250
)

img_res_monedas = img_original.copy()
conteo_monedas = {}

if circulos is not None:
    circulos = np.uint16(np.around(circulos))
    
    categorias = {
        "1 Peso (Bimetal)": [],
        "50 Centavos (Marrón)": [],
        "10 Centavos (Plateada)": [],
    }
    
    umbral_saturacion = 105

    for i in circulos[0, :]:
        x, y, r = i[0], i[1], i[2]

        # Máscaras para análisis de color 
        mask_centro = np.zeros(img_hsv.shape[:2], dtype="uint8")
        cv2.circle(mask_centro, (x, y), int(r * 0.40), 255, -1)

        mask_borde = np.zeros(img_hsv.shape[:2], dtype="uint8")
        cv2.circle(mask_borde, (x, y), int(r * 0.90), 255, -1)
        cv2.circle(mask_borde, (x, y), int(r * 0.60), 0, -1)

        mean_centro = cv2.mean(img_hsv, mask=mask_centro)[1] 
        mean_borde = cv2.mean(img_hsv, mask=mask_borde)[1]   

        # Clasificación
        if mean_centro > umbral_saturacion and mean_borde < umbral_saturacion:
            tipo = "1 Peso (Bimetal)"
        elif mean_centro < umbral_saturacion and mean_borde < umbral_saturacion:
            tipo = "10 Centavos (Plateada)"
        else:
            tipo = "50 Centavos (Marrón)"
        
        categorias[tipo].append(i)

    # Dibujar resultados monedas
    colores = {
        "1 Peso (Bimetal)": (0, 255, 0),       
        "50 Centavos (Marrón)": (0, 0, 255),   
        "10 Centavos (Plateada)": (255, 0, 0), 
    }

    for tipo, lista in categorias.items():
        conteo_monedas[tipo] = len(lista)
        for c in lista:
            cv2.circle(img_res_monedas, (c[0], c[1]), c[2], colores[tipo], 10)
            cv2.circle(img_res_monedas, (c[0], c[1]), 2, (0, 0, 0), 3)
            
    for k, v in conteo_monedas.items():
        print(f"{k}: {v}")

else:
    print("No se detectaron monedas.")






# ==========================================
# PARTE 2: DETECCIÓN Y CONTEO DE DADOS
# ==========================================

# Procesamiento morfológico
img_th = (img_blur < 20).astype(np.uint8) * 255
kernel_apertura = np.ones((9, 9), np.uint8)
img_open = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, kernel_apertura)

kernel_circular = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
img_dil = cv2.dilate(img_open, kernel_circular, iterations=1) # Puntos unificados para conteo

kernel_cierre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (65, 65))
img_close = cv2.morphologyEx(img_dil, cv2.MORPH_CLOSE, kernel_cierre) # Cuerpos de dados 

# Detección de contornos (cuerpos de los dados)
contornos, _ = cv2.findContours(img_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Dados encontrados: {len(contornos)}")

img_res_dados = img_original.copy()
total_puntos_dados = 0
lado_roi = 300

for contorno in contornos:
    x, y, w, h = cv2.boundingRect(contorno)
    cx, cy = x + (w // 2), y + (h // 2)

    mitad = lado_roi // 2
    p1_x, p1_y = max(0, cx - mitad), max(0, cy - mitad)
    p2_x, p2_y = min(img_dil.shape[1], cx + mitad), min(img_dil.shape[0], cy + mitad)

    # Recortar de la imagen dilatada (puntos)
    roi_puntos = img_dil[p1_y:p2_y, p1_x:p2_x]

    # Contar componentes 
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(roi_puntos, 4, cv2.CV_32S)
    valor_dado = num_labels - 1 
    total_puntos_dados += valor_dado

    # Dibujar resultados 
    cv2.rectangle(img_res_dados, (p1_x, p1_y), (p2_x, p2_y), (0, 255, 0), 6)
    cv2.putText(img_res_dados, str(valor_dado), (cx + 40, cy - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10, cv2.LINE_AA)

print(f"Suma total puntos dados: {total_puntos_dados}")





# ==========================================
# VISUALIZACIÓN FINAL
# ==========================================
plt.figure(figsize=(15, 7))

# --- Subplot 1: Monedas ---
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img_res_monedas, cv2.COLOR_BGR2RGB))
plt.title('Clasificación de Monedas')
plt.axis('off')

texto_stats = "Conteo\n"
for k, v in conteo_monedas.items():
    texto_stats += f"{k}: {v}\n"

plt.text(10, 50, texto_stats, 
         fontsize=10, 
         color='black',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))


# --- Subplot 2: Dados ---
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_res_dados, cv2.COLOR_BGR2RGB))
plt.title(f'Conteo de Dados (Suma: {total_puntos_dados})')
plt.axis('off')

plt.tight_layout()
plt.show()