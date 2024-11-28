import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector

# Inicializar detector
detector = HandDetector(detectionCon=0.8, maxHands=1)
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: No se pudo acceder a la cámara.")
    exit()

print("Presiona 'q' para salir. Presiona 'c' para borrar el texto.")

# Diccionario de lenguaje de señas para todas las letras (ASL)
# Aquí tenemos una representación simplificada de las letras
sign_language_dict = {
    'A': [1, 0, 0, 0, 0],  # Mano en forma de "A"
    'B': [1, 1, 0, 0, 0],  # Mano en forma de "B"
    'C': [1, 0, 0, 0, 1],  # Mano en forma de "C"
    'D': [1, 1, 0, 1, 0],  # Mano en forma de "D"
    'E': [1, 0, 0, 1, 0],  # Mano en forma de "E"
    'F': [1, 1, 1, 0, 0],  # Mano en forma de "F"
    'G': [1, 1, 1, 1, 0],  # Mano en forma de "G"
    'H': [1, 1, 0, 1, 1],  # Mano en forma de "H"
    'I': [1, 0, 1, 0, 0],  # Mano en forma de "I"
    'J': [1, 0, 1, 0, 1],  # Mano en forma de "J"
    'K': [1, 1, 1, 0, 1],  # Mano en forma de "K"
    'L': [1, 0, 1, 1, 0],  # Mano en forma de "L"
    'M': [1, 0, 0, 0, 0],  # Mano en forma de "M"
    'N': [1, 0, 1, 1, 1],  # Mano en forma de "N"
    'O': [1, 1, 1, 1, 1],  # Mano en forma de "O"
    'P': [1, 1, 1, 1, 0],  # Mano en forma de "P"
    'Q': [1, 0, 1, 1, 0],  # Mano en forma de "Q"
    'R': [1, 1, 0, 0, 1],  # Mano en forma de "R"
    'S': [1, 0, 0, 1, 1],  # Mano en forma de "S"
    'T': [0, 1, 0, 0, 1],  # Mano en forma de "T"
    'U': [0, 1, 0, 1, 0],  # Mano en forma de "U"
    'V': [0, 1, 1, 1, 0],  # Mano en forma de "V"
    'W': [1, 0, 0, 1, 1],  # Mano en forma de "W"
    'X': [0, 1, 0, 0, 0],  # Mano en forma de "X"
    'Y': [0, 1, 1, 0, 0],  # Mano en forma de "Y"
    'Z': [0, 0, 1, 0, 0],  # Mano en forma de "Z"
}

recognized_text = ""  # Texto acumulado

while True:
    ret, frame = video.read()
    if not ret:
        print("No se pudo capturar el cuadro de video.")
        break

    # Procesar la detección de manos
    hands, processed_frame = detector.findHands(frame, flipType=False)

    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)  # Detectar los dedos levantados

        # Detectar la orientación de la mano (palma o dorso)
        palm_side = hand['type']  # 'Right' o 'Left' de la mano detectada
        if palm_side == 'Left':
            # Si es la mano izquierda, invertimos el frame
            frame = cv2.flip(frame, 1)

        # Convertir la lista de dedos levantados a una tupla
        finger_tuple = tuple(fingers)

        # Reconocer la letra
        for letter, pattern in sign_language_dict.items():
            if finger_tuple == tuple(pattern):  # Comparar con el patrón de la letra
                recognized_text += letter  # Agregar letra al texto
                print(f"Letra detectada: {letter}")
                break  # Detenerse al encontrar la letra

        # Mostrar texto acumulado en pantalla
        cv2.putText(processed_frame, recognized_text, (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Mostrar el cuadro procesado
    cv2.imshow("Lenguaje de señas", processed_frame)

    # Si se presiona la tecla 'q', salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Si se presiona la tecla 'c', borrar el texto
    if cv2.waitKey(1) & 0xFF == ord('c'):
        recognized_text = ""  # Limpiar el texto

# Liberar recursos
video.release()
cv2.destroyAllWindows()
