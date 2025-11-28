# 🧠 Q-Learning para Navegación en Laberintos

Implementación de un agente de Aprendizaje por Refuerzo (Reinforcement Learning) que aprende a navegar laberintos usando el algoritmo **Q-Learning**.

## 📋 Descripción

Este proyecto implementa un agente inteligente capaz de aprender políticas óptimas para encontrar la salida en laberintos de diferentes complejidades. El agente utiliza Q-Learning, un algoritmo de aprendizaje por refuerzo libre de modelo, para descubrir rutas eficientes mediante la exploración y explotación del entorno.

### Características Principales

- ✅ Algoritmo Q-Learning con decay de epsilon
- ✅ Soporte para laberintos personalizados
- ✅ Múltiples salidas posibles
- ✅ Visualización interactiva del laberinto
- ✅ Generación de GIFs con la trayectoria del agente
- ✅ Guardado/carga de políticas aprendidas
- ✅ Métricas de entrenamiento (recompensas, pasos)

## 🛠️ Requisitos

### Python

- Python 3.10 o superior

### Librerías

```bash
pip install numpy pandas matplotlib pillow jupyter
```

O instalar desde el archivo de requisitos:

```bash
pip install -r requirements.txt
```

## 📁 Estructura del Proyecto

```
proyecto/
│
├── maze.ipynb                  # Notebook principal
├── utils.py                    # Funciones auxiliares (generación de GIFs)
├── README.md                   # Este archivo
├── requirements.txt            # Dependencias
│
├── laberintos/                 # Archivos de configuración de laberintos
│   ├── project_lab_v2.txt     # Laberinto 8×7 (recomendado)
│   ├── complex_12x10.txt      # Laberinto 12×10 (complejo)
│   ├── laberinto_8x7_v2.txt   # Variante alternativa
│   └── ...
│
├── qtables/                    # Políticas guardadas durante el entrenamiento
│   ├── qtable_policy100.json
│   ├── qtable_policy1000.json
│   └── ...
│
└── *.gif                       # Animaciones generadas
```

## 🚀 Uso

### 1. Preparación

Clona el repositorio y navega al directorio:

```bash
cd proyecto
```

### 2. Ejecutar el Notebook

Inicia Jupyter Notebook:

```bash
jupyter notebook maze.ipynb
```

### 3. Configurar Hiperparámetros

En la celda correspondiente, ajusta los parámetros según tus necesidades:

```python
# Hiperparámetros
EPISODES = 3000              # Número de episodios de entrenamiento
ALPHA = 0.15                 # Tasa de aprendizaje (learning rate)
GAMMA = 0.99                 # Factor de descuento (discount factor)
EPSILON_START = 0.9          # Exploración inicial
EPSILON_MIN = 0.01           # Exploración mínima
EPSILON_DECAY = 0.99         # Decaimiento de epsilon por episodio
MAX_STEPS_EPISODE = 50       # Máximo de pasos por episodio
MAZE = 'laberintos/project_lab_v2.txt'  # Archivo del laberinto
ALL_ACTIONS = True           # Considerar acciones inválidas
SAVE_BEST_POLICY = True      # Guardar la mejor política durante el entrenamiento
BEST_POLICY_FILE = "qtables/best_qtable_policy.json"
```

### 4. Entrenar el Agente

Ejecuta las celdas secuencialmente:

```python
# 1. Crear entorno
env = GridWorld(MAZE)
print('Inicio:', env.start_state, 'Meta:', env.goal_states)
env.visualize()

# 2. Inicializar agente
agent = QLearningAgent(env, alpha=ALPHA, gamma=GAMMA, 
                       epsilon=EPSILON_START, all_actions=ALL_ACTIONS)

# 3. Entrenar
episode_rewards, steps_per_episode = agent.run(
    episodes=EPISODES, 
    max_steps_per_episode=MAX_STEPS_EPISODE,
    epsilon_decay=EPSILON_DECAY,
    epsilon_min=EPSILON_MIN
)

# 4. Visualizar resultados
agent.plot_learning_curve(episode_rewards, steps_per_episode)
policy, values = agent.test_performance()
agent.plot_actions(policy, values)
```

### 5. Generar Animación (GIF)

```python
# Cargar política entrenada
env_video = GridWorld(MAZE)
agent_video = QLearningAgent(env_video, epsilon=0.0, all_actions=ALL_ACTIONS)
agent_video.load_policy('qtables/qtable_policy1000.json')

# Generar GIF
frames = generate_video_with_qvalues(
    agent_video, 
    output_filename='maze_solution.gif', 
    fps=2,
    max_steps=MAX_STEPS_EPISODE
)

# Visualizar en el notebook
from IPython.display import Image as IPImage, display
display(IPImage(filename='maze_solution.gif'))
```

## 📊 Formato de Archivos de Laberinto

Los laberintos se definen en archivos `.txt` con el siguiente formato:

```
n m          # Línea 1: Dimensiones (filas columnas)
k            # Línea 2: Número de segmentos de pared
x1 y1 x2 y2  # Líneas 3 a k+2: Coordenadas de cada segmento
x1 y1 x2 y2
...
```

### Ejemplo (laberinto 8×7):

```
8 7
65
0 0 0 1
0 1 0 2
...
8 6 8 7
```

### Coordenadas

- Las coordenadas son de **vértices**, no celdas
- Rango de filas: `0` a `n` (vértices)
- Rango de columnas: `0` a `m` (vértices)
- Las celdas están entre vértices consecutivos

### Entrada y Salida

- **Entrada**: Automáticamente detectada (columna 0 sin muro izquierdo)
- **Salidas**: Automáticamente detectadas (columna m-1 sin muro derecho)
- Puede haber múltiples salidas

## 🎯 Algoritmo Q-Learning

El agente aprende la función Q(s,a) que estima el valor de tomar la acción `a` en el estado `s`:

```
Q(s,a) ← Q(s,a) + α [r + γ max_{a'} Q(s',a') - Q(s,a)]
```

Donde:
- **α** (alpha): Tasa de aprendizaje
- **γ** (gamma): Factor de descuento
- **r**: Recompensa inmediata
- **s'**: Siguiente estado
- **ε** (epsilon): Probabilidad de exploración

### Esquema de Recompensas

| Evento | Recompensa |
|--------|------------|
| Alcanzar la meta | +100 |
| Movimiento válido | -1 |
| Colisión con pared/borde | -5 |

## 📈 Resultados

El notebook genera automáticamente:

1. **Curvas de aprendizaje**: Evolución de recompensas y pasos por episodio
2. **Visualización de la política**: Laberinto con flechas de acción y valores Q
3. **GIF animado**: Trayectoria del agente paso a paso con valores Q
4. **Archivos JSON**: Políticas guardadas cada 100 episodios

### Archivos Generados

- `qtables/qtable_policy{N}.json`: Política en el episodio N
- `maze_from_qtable.gif`: Animación de la solución

## 🔧 Personalización

### Crear tu Propio Laberinto

1. Crea un archivo `.txt` en `laberintos/` siguiendo el formato descrito
2. Define las dimensiones y los segmentos de pared
3. Actualiza la variable `MAZE` con la ruta de tu archivo
4. Ejecuta el entrenamiento

### Ajustar Comportamiento del Agente

- **Más exploración**: Aumenta `EPSILON_START` o reduce `EPSILON_DECAY`
- **Aprendizaje más rápido**: Aumenta `ALPHA` (cuidado con inestabilidad)
- **Planificación a largo plazo**: Aumenta `GAMMA` (cerca de 1.0)

## 📝 Notas Técnicas

### Detección Automática de Entrada/Salida

El código detecta automáticamente:
- **Entrada**: Primera celda en columna 0 sin muro izquierdo
- **Salidas**: Todas las celdas en columna m-1 sin muro derecho

### Consideración de Acciones Inválidas

- `ALL_ACTIONS = True`: El agente considera todas las acciones (up, down, left, right) incluso si hay pared
- `ALL_ACTIONS = False`: Solo considera acciones válidas (sin pared)

**Recomendación**: Usar `True` para convergencia más estable

## 🐛 Solución de Problemas

### Error: "El archivo debe tener al menos X líneas"

- **Causa**: El número de segmentos declarado no coincide con las líneas del archivo
- **Solución**: Verifica que la línea 2 del archivo coincida con el número real de segmentos

### El agente no converge

- Aumenta `EPISODES` (ej. 5000)
- Ajusta `ALPHA` (prueba valores entre 0.1 y 0.3)
- Verifica que el laberinto tenga solución

### GIF no se genera

- Asegúrate de que la política esté cargada correctamente
- Verifica que `utils.py` esté en el mismo directorio
- Revisa que PIL (Pillow) esté instalado

## 📚 Referencias

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
- Curso Aprendizaje por refuerzo (2025). Maestria Inteligencia artificial MaIA - Universidad de los Andes

## 👥 Autor

Proyecto desarrollado para el curso de Aprendizaje por Refuerzo - Universidad de los Andes - Antonio Miguel Porcelli Sposaro - Dubán Leandro Jiménez Puerta 

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.