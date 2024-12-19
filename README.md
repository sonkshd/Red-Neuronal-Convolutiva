Red Neuronal Convolutiva
Entrenamiento de una Red Neuronal Convolutiva para la clasificación de frutas utilizando un dataset de Kaggle.

Descarga del Dataset
Para utilizar el mismo dataset que este proyecto, puedes descargarlo directamente desde Kaggle con la librería kagglehub usando el siguiente código:

```python
import kagglehub
path = kagglehub.dataset_download("moltean/fruits")
print("Path to dataset files:", path)
```
------------------------------------------------------------------------------------------------------------------------------------------------------------
División del Dataset
En este proyecto, no se utiliza el dataset completo para el entrenamiento y el testeo. En su lugar, se realiza una división en proporción 80/20:

El 80% del dataset se utiliza para el entrenamiento.
El 20% restante se utiliza para la evaluación (testeo).
La división se realiza con el siguiente fragmento de código:

```python
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
```
------------------------------------------------------------------------------------------------------------------------------------------------------------
Arquitectura de la Red Neuronal
El modelo está compuesto por tres capas convolutivas. Cada capa toma en cuenta el formato de entrada RGB de las imágenes, por lo que se configuran tres canales de entrada. Si se trabajara con imágenes en escala de grises, sería necesario ajustar los canales a 1.

Se implementa una técnica de regularización mediante Dropout para reducir el sobreajuste (overfitting). En este caso, se desactiva el 30% de las neuronas de manera aleatoria, ya que valores superiores, como el 50%, resultaron en una disminución del rendimiento.

El diseño de las capas principales es el siguiente:

```python
self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
self.dropout = nn.Dropout(0.3)
```
------------------------------------------------------------------------------------------------------------------------------------------------------------
Inspección de Tensores
Es importante verificar la forma de los tensores generados tras las transformaciones aplicadas por las capas convolutivas. La estructura esperada del tensor sigue el siguiente esquema:


(batch_size, número_de_canales, alto_imagen, ancho_imagen)
El siguiente fragmento de código permite inspeccionar la forma de los tensores usando un lote de datos:

```python
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# Inspeccionar el primer lote
data_iter = iter(dataloader)
images, labels = next(data_iter)
print(f"Forma del tensor: {images.shape}")
```
------------------------------------------------------------------------------------------------------------------------------------------------------------
Notas Adicionales
Este modelo está optimizado para clasificar imágenes RGB.
La configuración del Dropout y el número de capas convolutivas pueden ajustarse según la naturaleza del dataset y los resultados deseados.
El proyecto se puede ampliar utilizando técnicas avanzadas como data augmentation o arquitecturas más profundas para mejorar la precisión del modelo.
