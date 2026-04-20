import torch

import active_adaptation
from active_adaptation.envs.base import _Env

class SimpleEnv(_Env):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.robot = self.scene["robot"]

    def setup_scene(self):
        if active_adaptation.get_backend() != "mjlab":
            raise NotImplementedError(
                f"Unsupported backend: {active_adaptation.get_backend()}"
            )

        from mjlab.scene import SceneCfg as MJSceneCfg
        from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
        from mjlab.terrains import TerrainEntityCfg
        import mjlab.terrains as terrain_gen
        from mjlab.sensor import ContactMatch, ContactSensorCfg
        from mjlab.sim import MujocoCfg, SimulationCfg
        from mjlab.scene import Scene
        from mjlab.sim.sim import Simulation

        mjlab_dt = self.cfg.sim.get("mjlab_physics_dt", None)
        if mjlab_dt is None:
            mjlab_dt = self.cfg.sim.get("mujoco_physics_dt", None)
        decimation_est = max(1, int(round(float(self.cfg.sim.step_dt) / float(mjlab_dt))))

        env_spacing = self.cfg.viewer.get("env_spacing", 2.5)
        scene_cfg = MJSceneCfg(num_envs=self.cfg.num_envs, env_spacing=env_spacing)

        scene_cfg.terrain = TerrainEntityCfg(
            terrain_type="plane",
            env_spacing=env_spacing,
            num_envs=self.cfg.num_envs,
        )

        from active_adaptation.assets import (
            get_robot_cfg,
            get_tennis_ball_cfg,
            get_tennis_court_cfg,
        )

        scene_cfg.entities["robot"] = get_robot_cfg(self.cfg.robot.name)
        tennis_cfg = self.cfg.get("tennis", None)
        if tennis_cfg is not None:
            if bool(tennis_cfg.get("add_court", False)):
                court_texture = str(tennis_cfg.get("court_texture", "green"))
                net_height = float(tennis_cfg.get("net_height", 1.07))
                net_collision_half_thickness = float(tennis_cfg.get("net_collision_half_thickness", 0.06))
                enable_racket_court_collision = bool(tennis_cfg.get("racket_court_collision", False))
                scene_cfg.entities["tennis_court"] = get_tennis_court_cfg(
                    texture=court_texture,
                    net_height=net_height,
                    net_collision_half_thickness=net_collision_half_thickness,
                    enable_racket_court_collision=enable_racket_court_collision,
                )
            if bool(tennis_cfg.get("add_ball", False)):
                scene_cfg.entities["tennis_ball"] = get_tennis_ball_cfg()

        # contact_cfg = ContactSensorCfg(
        #     name="contact_forces",
        #     primary=ContactMatch(mode="subtree", pattern=r".*", entity="robot"),
        #     secondary=ContactMatch(mode="body", pattern="terrain"),
        #     fields=("found", "force"),
        #     reduce="netforce",
        #     num_slots=1,
        #     track_air_time=True,
        #     history_length=3
        # )

        contact_cfg = ContactSensorCfg(
            name="contact_forces",
            primary=ContactMatch(
                mode="subtree",
                pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            global_frame=True,
            num_slots=1,
            track_air_time=True,
            history_length=3,
            debug=False
        )

        sensors = [contact_cfg]
        if (
            tennis_cfg is not None
            and bool(tennis_cfg.get("add_ball", False))
            and bool(tennis_cfg.get("add_court", False))
            and bool(tennis_cfg.get("enable_contact_sensors", True))
        ):
            racket_ball_sensor = ContactSensorCfg(
                name="racket_ball_contact",
                primary=ContactMatch(mode="geom", pattern="tennis_racket_collision", entity="robot"),
                secondary=ContactMatch(mode="geom", pattern="tennis_ball_geom", entity="tennis_ball"),
                fields=("found", "force"),
                reduce="maxforce",
                global_frame=False,
                num_slots=1,
                history_length=decimation_est,
                debug=False,
            )
            ball_net_sensor = ContactSensorCfg(
                name="ball_net_contact",
                primary=ContactMatch(mode="geom", pattern="tennis_ball_geom", entity="tennis_ball"),
                secondary=ContactMatch(mode="geom", pattern="tennis_net_collision", entity="tennis_court"),
                fields=("found", "force"),
                reduce="maxforce",
                global_frame=False,
                num_slots=1,
                history_length=decimation_est,
                debug=False,
            )
            ball_court_sensor = ContactSensorCfg(
                name="ball_court_contact",
                primary=ContactMatch(mode="geom", pattern="tennis_ball_geom", entity="tennis_ball"),
                secondary=ContactMatch(mode="geom", pattern="tennis_court_ball_collision", entity="tennis_court"),
                fields=("found", "force"),
                reduce="maxforce",
                global_frame=False,
                num_slots=1,
                history_length=decimation_est,
                debug=False,
            )
            sensors.extend([racket_ball_sensor, ball_net_sensor, ball_court_sensor])
            if bool(tennis_cfg.get("racket_court_collision", False)):
                racket_court_sensor = ContactSensorCfg(
                    name="racket_court_contact",
                    primary=ContactMatch(mode="geom", pattern="tennis_racket_collision", entity="robot"),
                    secondary=ContactMatch(mode="geom", pattern="tennis_court_racket_collision", entity="tennis_court"),
                    fields=("found", "force"),
                    reduce="maxforce",
                    global_frame=False,
                    num_slots=1,
                    history_length=decimation_est,
                    debug=False,
                )
                sensors.append(racket_court_sensor)

        scene_cfg.sensors = tuple(sensors)

        self.sim_cfg = sim_cfg = SimulationCfg(
            nconmax=200,
            njmax=500,
            mujoco=MujocoCfg(
                timestep=mjlab_dt,
                iterations=10,
                ls_iterations=20,
            ),
        )

        self.scene = Scene(scene_cfg, device=self.device)
        self.sim = Simulation(
            num_envs=self.scene.num_envs,
            cfg=sim_cfg,
            model=self.scene.compile(),
            device=self.device,
        )

        self.scene.initialize(
            mj_model=self.sim.mj_model,
            model=self.sim.model,
            data=self.sim.data,
        )
        if not hasattr(self.scene, "env_origins") and hasattr(self.scene, "env_offsets"):
            self.scene.env_origins = self.scene.env_offsets
        self._align_tennis_court_to_env_origins()

    def _align_tennis_court_to_env_origins(self, env_ids: torch.Tensor | None = None):
        if "tennis_court" not in self.scene.entities:
            return
        court = self.scene["tennis_court"]
        if not getattr(court, "is_mocap", False):
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        pose = torch.zeros((env_ids.numel(), 7), device=self.device, dtype=torch.float32)
        pose[:, :3] = self.scene.env_origins[env_ids]
        pose[:, 3] = 1.0
        court.write_mocap_pose_to_sim(pose, env_ids=env_ids)

        
    def _reset_idx(self, env_ids: torch.Tensor):
        init_root_state = self.command_manager.sample_init(env_ids)
        if init_root_state is not None and not self.robot.is_fixed_base:
            self.robot.write_root_state_to_sim(init_root_state, env_ids=env_ids)
        self.stats[env_ids] = 0.0
        if hasattr(self.scene, "reset"):
            self.scene.reset(env_ids)
        self._align_tennis_court_to_env_origins(env_ids)

    def render(self, mode: str = "human"):
        return super().render(mode)
