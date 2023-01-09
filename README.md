# Proyecto RL - Training in Highway Environment

- Highway-env es un proyecto reúne una colección de entornos para la toma de decisiones en Conducción Autónoma.
- Tiene 6 entornos distintos para entrenar los modelos: Highway, Merge, Roundabout, Parking, Intersection, Racetrack.
- Cuenta con un método ágil de configuración del entorno, para poder modificar las observaciones, acciones y recompensas.
- Enlace a la fuente: https://highway-env.readthedocs.io/en/latest/index.html

## Instalacion

En el directorio principal se encuentra el fichero requirements.txt.

Ejecutar el siguiente comando:

```bash
pip install -r requirements.txt
```

## Roundabout-v0 PPO

Estos son los detalles del proyecto:

- El script de entrenamiento se encuentra en la carpeta train, con el mismo nombre del entorno.
- El script para probar los modelos se encuentra en la carpeta predict, con el mismo nombre del entorno.
- Los logs de este entorno se encuentran dentro de la carpeta logs, con este mismo nombre.
- Los modelos obtenidos en el entrenamiento se encuentran en la carpeta models, con el mismo nombre del entorno.
- PPO implementado por defecto, bajo los hiperparametros de stable-baseline.

Estos son los resultados y conclusiones extraidas:

- Conclusiones:
 - Recompensa media: 10,012
 - Desviación estándar de las recompensas: 0,244
 - Lento pero seguro

## Intersection-v0 DQN

Estos son los detalles del proyecto:

- El script de entrenamiento se encuentra en la carpeta train, con el mismo nombre del entorno.
- El script para probar los modelos se encuentra en la carpeta predict, con el mismo nombre del entorno.
- Los logs de este entorno se encuentran dentro de la carpeta logs, con este mismo nombre.
- Los modelos obtenidos en el entrenamiento se encuentran en la carpeta models, con el mismo nombre del entorno.
- DQN implementado por defecto, bajo los hiperparametros de stable-baseline.


Esta es la configuracion del entorno:

```python
env.configure({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                },
                "absolute": True,
                "flatten": False,
                "observe_intentions": False
            },
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True,
                "target_speeds": [0, 3, 9]
            },
            "duration": 40,  # [s]
            "destination": "o1",
            "controlled_vehicles": 1,
            "initial_vehicle_count": 3,
            "spawn_probability": 0.8,
            "screen_width": 500,
            "screen_height": 500,
            "centering_position": [0.5, 0.6],
            "scaling": 5.5 * 1.3,
            "collision_reward": -6,
            "high_speed_reward": 0,
            "arrived_reward": 2,
            "reward_speed_range": [0.0, 1.0],
            "normalize_reward": True,
            "offroad_terminal": False
})
```
Esta configuracion fue cambiada durante el entrenamiento, empezando con un spawn de vehiculos pequeño, duracion de 20 segundos, reward de colisión baja y alta por velocidad. La estrategia fue buscar aumentar poco a poco los coches, tratando de mantener la velocidad, ajustando las recompensas durante las paradas en las etapas de entrenamiento.


Estos son los resultados y conclusiones extraidas:

- Conclusiones:
 - Recompensa media: 2.0335285
 - Desviación estándar de las recompensas: 2.774574973403359
 - Difícil de entrenar, falta de consistencia en el entorno y en el agente.


## Highway-v0 PPO

Estos son los detalles del proyecto:

- El script de entrenamiento se encuentra en la carpeta train, con el mismo nombre del entorno.
- El script para probar los modelos se encuentra en la carpeta predict, con el mismo nombre del entorno.
- Los logs de este entorno se encuentran dentro de la carpeta logs, con este mismo nombre.
- Los modelos obtenidos en el entrenamiento se encuentran en la carpeta models, con el mismo nombre del entorno.

El modelo fue entrenado con los siguientes hiperparametros:

```python
model = PPO('MlpPolicy', 
            env,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=5e-4,
            batch_size=32,
            gamma=0.8,
            verbose=0,
            tensorboard_log="../logs/highway-fast_ppo/")
```

Esta es la configuracion del entorno:

```python
env.configure({
    "observation": {
        "type": "Kinematics"
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 5,
    "vehicles_count": 200,
    "duration": 60,  # [s]
    "initial_spacing": 2,
    "collision_reward": -4,  # The reward received when colliding with a vehicle.
    "reward_speed_range": [20, 40],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD]. subido de 30 a 40
    "simulation_frequency": 34,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 300,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False
})
```
Esta configuracion fue cambiada durante el entrenamiento, empezando con un spawn de vehiculos pequeño, duracion de 20 segundos, reward de colision baja y alta por velocidad. La estrategia fue buscar aumentar poco a poco los coches y las lineas de la carretera, tratando de mantener la velocidad, ajustando las recompensas durante las paradas en las etapas de entrenamiento.


Estos son los resultados y conclusiones extraidas:

- Conclusiones:
 - Recompensa media: 22.025547800000002
 - Desviación estándar de las recompensas: 1.9381078254715245
 - Lento pero seguro


## Highway-v0 DQN

Estos son los detalles del proyecto:

- El script de entrenamiento se encuentra en la carpeta train, con el mismo nombre del entorno.
- El script para probar los modelos se encuentra en la carpeta predict, con el mismo nombre del entorno.
- Los logs de este entorno se encuentran dentro de la carpeta logs, con este mismo nombre.
- Los modelos obtenidos en el entrenamiento se encuentran en la carpeta models, con el mismo nombre del entorno.

El modelo fue entrenado con los siguientes hiperparametros:

```python
model = DQN('MlpPolicy', 
            env,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=5e-4,
            buffer_size=15000,
            learning_starts=200,
            batch_size=32,
            gamma=0.8,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=50,
            verbose=1,
            tensorboard_log="logs/highway-fast_dqn-2L/")
```

Esta es la configuracion del entorno:

```python
env.configure({
    "observation": {
        "type": "Kinematics"
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 5,
    "vehicles_count": 200,
    "duration": 60,  # [s]
    "initial_spacing": 2,
    "collision_reward": -4,  # The reward received when colliding with a vehicle.
    "reward_speed_range": [20, 40],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD]. subido de 30 a 40
    "simulation_frequency": 34,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 300,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False
})
```
Esta configuracion fue cambiada durante el entrenamiento, empezando con un spawn de vehiculos pequeño, duracion de 20 segundos, reward de colision baja y alta por velocidad. La estrategia fue buscar aumentar poco a poco los coches y las lineas de la carretera, tratando de mantener la velocidad, ajustando las recompensas durante las paradas en las etapas de entrenamiento.


Estos son los resultados y conclusiones extraidas:

- Conclusiones:
 - Recompensa media: 20.578723000000004
 - Desviación estándar de las recompensas: 4.880833384757565
 - Rápido, buenos episodios pero con errores


## Highway-v0 DQN 3-Shared-Layers(values-action) 

Estos son los detalles del proyecto:

- El script de entrenamiento se encuentra en la carpeta train, con el mismo nombre del entorno.
- El script para probar los modelos se encuentra en la carpeta predict, con el mismo nombre del entorno.
- Los logs de este entorno se encuentran dentro de la carpeta logs, con este mismo nombre.
- Los modelos obtenidos en el entrenamiento se encuentran en la carpeta models, con el mismo nombre del entorno.

El modelo fue entrenado con los siguientes hiperparametros:

```python
model = DQN('MlpPolicy', 
            env,
            policy_kwargs=dict(net_arch=[256, 256, 256]),
            learning_rate=5e-4,
            buffer_size=20000,
            learning_starts=200,
            batch_size=64,
            gamma=0.8,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=50,
            verbose=1,
            tensorboard_log="logs/highway-fast_dqn-3L/")
```

Esta es la configuracion del entorno:

```python
env.configure({
    "observation": {
        "type": "Kinematics"
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 5,
    "vehicles_count": 200,
    "duration": 60,  # [s]
    "initial_spacing": 2,
    "collision_reward": -4,  # The reward received when colliding with a vehicle.
    "reward_speed_range": [20, 40],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD]. subido de 30 a 40
    "simulation_frequency": 34,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 300,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False
})
```
Esta configuracion fue cambiada durante el entrenamiento, empezando con un spawn de vehiculos pequeño, duracion de 20 segundos, reward de colision baja y alta por velocidad. La estrategia fue buscar aumentar poco a poco los coches y las lineas de la carretera, tratando de mantener la velocidad, ajustando las recompensas durante las paradas en las etapas de entrenamiento.


Estos son los resultados y conclusiones extraidas:

- Conclusiones:
 - Recompensa media: 20.5268218
 - Desviación estándar de las recompensas: 4.708001783433473
 - Lento, pocos adelantamientos y algunos fallos


## Merge-v0 DQN 

Estos son los detalles del proyecto:

- El script de entrenamiento se encuentra en la carpeta train, con el mismo nombre del entorno.
- El script para probar los modelos se encuentra en la carpeta predict, con el mismo nombre del entorno.
- Los logs de este entorno se encuentran dentro de la carpeta logs, con este mismo nombre.
- Los modelos obtenidos en el entrenamiento se encuentran en la carpeta models, con el mismo nombre del entorno.

El modelo fue entrenado con los siguientes hiperparametros:

```python
model = DQN('MlpPolicy', 
            env,
            policy_kwargs=dict(net_arch=[256, 256, 256]),
            learning_rate=5e-4,
            buffer_size=15000,
            learning_starts=200,
            batch_size=256,
            gamma=0.8,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=50,
            verbose=1,
            tensorboard_log="logs/merge_dqn/")
```


Estos son los resultados y conclusiones extraidas:

- Conclusiones:
 - Recompensa media: 12.971319
 - Desviación estándar de las recompensas: 0,0
 - Lento pero seguro