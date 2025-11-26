import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)



# crear lista con los nombres de las imagenes
nombres_imagenes = []
for i in range(1, 13):
    i = str(i)
    if len(i) != 1:
        nombre = f'img{i}.png'
    else:
        nombre = f'img0{i}.png'
    nombres_imagenes.append(nombre)



# Crear lista de  diccionarios con el nombre de la imagen, la imagen, y el recorte de la patente
imagenes = []
for nombre in nombres_imagenes:
    img = cv2.imread(nombre, cv2.IMREAD_GRAYSCALE)
    imagenes.append({
        'nombre': nombre,
        'img': img,
        'recorte_patente': None})



# Recorrer las imagenes para detectar la patente

for i in imagenes:
    # Recorta el 60% central de la imagen para eliminar un poco de ruido externo
    img = i['img']
    h, w = img.shape
    frac = 0.6
    nw = int(w*frac)
    nh = int(h*frac)
    xs = (w - nw)//2
    ys = (h - nh)//2
    img_c = img[ys:ys+nh, xs:xs+nw].copy()

    #imshow(img_c, title='imagen recortada')

    # Gradiente morfologico

    # Gradiente morfologico para detectar bordes
    kernel_gm = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    img_gm = cv2.morphologyEx(img_c, cv2.MORPH_GRADIENT, kernel_gm)

    #imshow(img_gm, title='gradiente morfologico')

    # Aplicar el filtro Gaussiano borrosidad
    img_blur = cv2.GaussianBlur(img_gm, (15, 15), 0)

    #imshow(img_blur, title='GaussianBlur')


    # Umbralado identificar las areas donde hay mas concentracion de blanco
    _, img_binary = cv2.threshold(img_blur, 110, 255, cv2.THRESH_BINARY)

    #imshow(img_binary, title='Imagen binaria')


    # Unir las areas horizontalmente mediante clausura
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
    closed = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel_horizontal)

    #imshow(closed, title='Clausura horizontal')



    # Eliminar lineas horizontales finas (como las de las parrillas de los autos)
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8))
    # Esto elimina líneas horizontales más finas que 8 píxeles de alto

    sin_lineas = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_vertical)

    #imshow(sin_lineas, title= 'Clausura vertical')


    
    # Encontrar formas rectangulares con relacion de aspecto aproximadamente 36/13 = 2.77 que es lo que miden las patentes
    # y se agrega al diccionario el recorte de la patente. 

    # --- parámetros configurables ---
    area_min = 500          # área mínima del contorno
    ratio_target = 2.77     # relación de aspecto ideal
    ratio_tol = 0.6        # tolerancia (2.77 ± 0.35)

    contours, _ = cv2.findContours(sin_lineas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []


    for c in contours:
        area = cv2.contourArea(c)
        if area < area_min:
            continue

        rect = cv2.minAreaRect(c)
        (w_rect, h_rect) = rect[1]

        if w_rect == 0 or h_rect == 0:
            continue

        ar = max(w_rect, h_rect) / min(w_rect, h_rect)

        # condición principal → forma rectangular con aspecto ~2.77
        if abs(ar - ratio_target) <= ratio_tol:
            candidatos.append(rect)


    
    if len(candidatos) == 0:
        print("No se encontró ningún rectángulo con ese aspecto.")
    else:
        # elegimos el que más se parece
        mejor = min(
            candidatos,
            key=lambda r: abs((max(r[1][0], r[1][1]) / min(r[1][0], r[1][1])) - ratio_target))

        # convertimos el rectángulo a 4 puntos
        box = cv2.boxPoints(mejor)
        box = box.astype(np.int32)

        # recuperar coordenadas respecto a la imagen original
        box[:,0] += xs
        box[:,1] += ys

        # dibujar
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        _ = cv2.drawContours(img_color, [box], -1, (0,255,0), 1)


        # Convertir bounding box a rectángulo axis-aligned para recorte
        # Convertir bounding box a rectángulo axis-aligned para recorte
        x, y, w_rect, h_rect = cv2.boundingRect(box)

        # Recorte de la patente desde la imagen original
        crop = img[y:y+h_rect, x:x+w_rect].copy()

        # Guardar en el diccionario de esta imagen
        i['recorte_patente'] = crop
        i['coords_patente'] = (x, y, w_rect, h_rect)

        #imshow(img_color)


for i in imagenes:
    nombre = i['nombre']
    img = i['img'] # Esta es la imagen original en escala de grises
    patente = i['recorte_patente']
    
    # Recuperamos las coordenadas de la patente en la imagen original
    # x_plate e y_plate son el "offset" o desplazamiento
    (x_plate, y_plate, w_plate, h_plate) = i['coords_patente']
    
    th = 80
    encontrado = False

    while th < 230: 
        _, patente_thresh = cv2.threshold(patente, th, 255, cv2.THRESH_BINARY)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(patente_thresh, connectivity=8)
        
        candidatos = []
        
        for k in range(1, num_labels):
            x, y, w, h, area = stats[k]
            
            if area < 20: continue 

            aspect_ratio = h / w

            if 1.5 < aspect_ratio < 3.5:
                candidatos.append(stats[k])

        # Si encontramos exactamente 6 caracteres
        if len(candidatos) == 6:
            
            # 1. Convertir la imagen ORIGINAL a color para poder dibujar sobre ella
            img_resultado = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # 2. Dibujar el bounding box de la PATENTE (en Rojo por ejemplo)
            _ = cv2.rectangle(img_resultado, 
                          (x_plate, y_plate), 
                          (x_plate + w_plate, y_plate + h_plate), 
                          (0, 0, 255), 2) # Rojo, grosor 2

            # 3. Dibujar los bounding boxes de los CARACTERES (en Verde)
            for (xc, yc, wc, hc, area) in candidatos:
                # TRANSFORMACIÓN DE COORDENADAS
                # Coordenada Global = Coordenada Patente + Coordenada Carácter
                x_global = x_plate + xc
                y_global = y_plate + yc
                
                _ = cv2.rectangle(img_resultado, 
                              (x_global, y_global), 
                              (x_global + wc, y_global + hc), 
                              (0, 255, 0), 1) # Verde, grosor 1
            
            # Mostrar la imagen original completa con todos los boxes
            imshow(img_resultado, title=f"Resultado Final: {nombre}") 
            encontrado = True
            break 
        
        th += 10

    if not encontrado:
        print(f"Patente {nombre}: No se pudieron reconocer los 6 caracteres.")


plt.show()