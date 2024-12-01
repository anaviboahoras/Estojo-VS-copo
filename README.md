# Estojo-VS-copo
# Nome do Projeto: Rede Neural Artificial com Database
# Objetivo: Criar o meu primeiro DataBase + uma RNA

# Descrição:
Projeto de criação de uma rede neural para reconhecer objetos ou animais.
Apartir de um database criado por nós alunos da DIO e BairesDev.
O meu alvo de pesquisa foi um estojo rosa, e um copo laranja.

## Funcionalidades

- Identificar a classe um = Copo 0
- Identificar a classe dois = Estojo 1

## Meu principal objetivo e motivação
É porque é o príncipio de como aplicar esse código para uma IA, reconhecer rostos, imagens.
Com o objetivo de auxiliar e automatizar exames médicos,cibersegurança e encontrar padrões
em diversas áreas do conhecimento.Por isso,esse projeto(que é o meu primeiro)foi tão importante
para mim,tanto por conta da experiência quanto dessass aplicações que podem ser inseridos.

Código

import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2
import imutils

## Desativar notação científica para maior clareza
np.set_printoptions(suppress=True)

## Função para carregar o modelo
def carregar_modelo(caminho_modelo):
return load_model(caminho_modelo, compile=False)

Labels diretamente no código
labels = [
"0 Copo",
"1 Estojo"
]

## Função principal de previsão
def fazer_previsao(frame, modelo, labels):
image = imutils.resize(frame, width=224, height=224)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = Image.fromarray(image)

size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
data[0] = normalized_image_array

prediction = modelo.predict(data, verbose=0)
index = np.argmax(prediction)
class_name = labels[index]
confidence_score = prediction[0][index]

return class_name, confidence_score
Caminhos dos arquivos
model_path = "C:/keras_model.h5"

## Carregar modelo
modelo = carregar_modelo(model_path)

## Exibir conteúdo das labels para verificar
print("Conteúdo das labels:")
for label in labels:
print(label.strip())

## Inicializar a captura de vídeo
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
ret, frame = cap.read()
if not ret:
break

class_name, confidence_score = fazer_previsao(frame, modelo, labels)

text = f"Class: {class_name[2:].strip()}, Confidence: {confidence_score:.2f}"
frame = cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

cv2.imshow('Prediction', frame)
key = cv2.waitKey(1)
if key == ord('q'):
    break
## Adicionar estas linhas para fechar a câmera e liberar o recurso adequadamente
cap.release()
cv2.destroyAllWindows()
