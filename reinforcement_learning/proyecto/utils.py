import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io


def generate_video_with_qvalues(agent, output_filename='agent_maze.gif', fps=2, max_steps=100):
    """
    Genera un video GIF mostrando la trayectoria del agente en el laberinto,
    incluyendo los valores Q en cada paso.    

    Args:
        agent: QLearningAgent con Q-table entrenada
        output_filename: nombre del archivo de salida (.gif)
        fps: frames por segundo (menor = más lento)
        max_steps: máximo de pasos a ejecutar

    Returns:
        frames: lista de imágenes PIL generadas
    """

    # Ejecutar política para obtener trayectoria
    state = agent.env.reset()
    trajectory = [state]
    actions_taken = []

    for _ in range(max_steps):
        action = agent.best_action(state)
        actions_taken.append(action)
        next_state, reward, done, _ = agent.step(state, action)
        trajectory.append(next_state)
        state = next_state
        if done:
            break

    print(
        f"Trayectoria: {len(trajectory)} estados, {len(actions_taken)} acciones")

    # Lista para almacenar frames
    frames = []

    def draw_maze_base(ax):
        """Dibuja elementos estáticos del laberinto"""
        ax.clear()

        # Cuadrícula
        for i in range(agent.env.lab.n + 1):
            ax.plot([0, agent.env.lab.m], [i, i], 'lightgray', linewidth=0.5)
        for j in range(agent.env.lab.m + 1):
            ax.plot([j, j], [0, agent.env.lab.n], 'lightgray', linewidth=0.5)

        # Paredes
        for (x1, y1, x2, y2) in agent.env.lab.paredes:
            ax.plot([y1, y2], [x1, x2], 'black', linewidth=3)

        # Celda de inicio (amarillo)
        start_row, start_col = agent.env.start_state
        ax.add_patch(patches.Rectangle((start_col, start_row), 1, 1,
                                       facecolor='#F6D924', alpha=0.3, edgecolor='black', linewidth=1))

        # Celda meta (verde)
        for goal_row, goal_col in agent.env.goal_states:
            ax.add_patch(patches.Rectangle((goal_col, goal_row), 1, 1,
                                           facecolor='#68FF33', alpha=0.6, edgecolor='black', linewidth=1))
            ax.text(goal_col + 0.5, goal_row + 0.5, 'META',
                ha='center', va='center', fontsize=10, color='darkgreen', weight='bold')

    def draw_qvalues(ax, state):
        """Dibuja los valores Q de las acciones posibles desde el estado actual"""
        row, col = state
        possible_actions = agent.env.get_possible_actions(state, agent.all_actions)

        action_positions = {
            'up': (col + 0.5, row + 0.15),
            'down': (col + 0.5, row + 0.85),
            'left': (col + 0.15, row + 0.5),
            'right': (col + 0.85, row + 0.5)
        }

        action_symbols = {
            'up': '↑',
            'down': '↓',
            'left': '←',
            'right': '→'
        }

        ax.add_patch(patches.Rectangle((col, row), 1, 1,
                                       facecolor='lightblue', alpha=0.3,
                                       edgecolor='blue', linewidth=2))

        for action in possible_actions:
            q_value = agent.get_value(state, action)
            pos_x, pos_y = action_positions[action]
            symbol = action_symbols[action]

            if q_value > 0:
                color = 'green'
            elif q_value < 0:
                color = 'red'
            else:
                color = 'gray'

            ax.text(pos_x, pos_y, f'{symbol}\n{q_value:.1f}',
                    ha='center', va='center', fontsize=8,
                    color=color, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=color, alpha=0.8))

    def draw_agent(ax, state, is_last=False):
        """Dibuja el agente en su posición actual"""
        row, col = state
        color = 'red' if is_last else 'blue'
        ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.2,
                                color=color, alpha=0.8, zorder=10))
        ax.text(col + 0.5, row + 0.5, 'A',
                ha='center', va='center', fontsize=12,
                color='white', weight='bold', zorder=11)

    def draw_path(ax, current_step):
        """Dibuja la trayectoria recorrida hasta el paso actual"""
        if current_step > 0:
            path_so_far = trajectory[:current_step+1]
            rows = [p[0] + 0.5 for p in path_so_far]
            cols = [p[1] + 0.5 for p in path_so_far]
            ax.plot(cols, rows, 'b--', linewidth=2,
                    alpha=0.5, label='Trayectoria')

    # Generar cada frame
    print(f"Generando {len(trajectory)} frames...")
    for frame_idx in range(len(trajectory)):
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        draw_maze_base(ax)

        current_state = trajectory[frame_idx]
        is_last = (frame_idx == len(trajectory) - 1)

        draw_path(ax, frame_idx)

        if not is_last:
            #print(agent.all_actions)
            draw_qvalues(ax, current_state)

        draw_agent(ax, current_state, is_last)

        if frame_idx < len(actions_taken):
            action_taken = actions_taken[frame_idx]
            ax.set_title(f'Paso {frame_idx+1}/{len(trajectory)-1} | Acción: {action_taken} | Estado: {current_state}',
                         fontsize=12, weight='bold')
        else:
            ax.set_title(f'¡META ALCANZADA! | Total pasos: {len(actions_taken)}',
                         fontsize=12, weight='bold', color='green')

        ax.set_xlim(-0.5, agent.env.lab.m + 0.5)
        ax.set_ylim(-0.5, agent.env.lab.n + 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img.copy())
        buf.close()
        plt.close(fig)

    # Guardar GIF
    print(f"Guardando GIF en {output_filename}...")
    duration = int(1000 / fps)
    frames[0].save(
        output_filename,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    print(f"✓ GIF guardado exitosamente: {output_filename}")
    print(f"  Tamaño: {len(frames)} frames")
    print(f"  Duración por frame: {duration}ms")

    return frames
