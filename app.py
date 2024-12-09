import cv2
import os
import numpy as np
import face_recognition
import streamlit as st

def load_known_faces():
    faces_path = "faces"
    facesEncodinds = []
    facesNames = []

    for file_name in os.listdir(faces_path):
        filePath = os.path.join(faces_path, file_name)
        image = cv2.imread(filePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        f_coding = face_recognition.face_encodings(image)[0]
        facesEncodinds.append(f_coding)
        facesNames.append(file_name.split(".")[0])
    
    return facesEncodinds, facesNames

def main():
    st.title("Reconhecimento Facial em Tempo Real")
    st.logo(image="https://unifametro.edu.br/wp-content/uploads/2024/11/UNIFAMETRO-Favicon.png",
        icon_image="https://unifametro.edu.br/wp-content/uploads/2024/11/UNIFAMETRO-Favicon.png")

    # Carrega rostos já "conhecidos"
    try:
        facesEncodinds, facesNames = load_known_faces()
    except Exception as e:
        st.error(f"Erro ao carregar rostos: {e}")
        return

    # Usa Haar Cascade, incluído com o OpenCV
    cascade_path = "haarcascade_frontalface_default.xml"
    faceClassificador = cv2.CascadeClassifier(cascade_path)

    # Cria uma parte pra exibir o vídeo
    stframe = st.empty()

    # Inicia webcam
    cap = cv2.VideoCapture(0)

    # Cria checkbox
    run = st.checkbox("Iniciar Reconhecimento Facial")

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Não foi possível acessar a webcam.")
            break
        
        frame = cv2.flip(frame, 1)
        orig = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceClassificador.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            face = orig[y:y + h, x:x + w]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            
            try:
                atual_face_encoding = face_recognition.face_encodings(face, known_face_locations=[(0, w, h, 0)])[0]
                result = face_recognition.compare_faces(facesEncodinds, atual_face_encoding)

                if True in result:
                    index = result.index(True)
                    name = facesNames[index]
                    color = (125, 220, 0)
                else:
                    name = "Desconhecido"
                    color = (50, 50, 255)
            except IndexError:
                name = "Sem rosto detectado"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y + h), (x + w, y + h + 30), color, -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y + h + 25), 2, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Converte o frame para exibição no Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")

    cap.release()

if __name__ == "__main__":
    main()
