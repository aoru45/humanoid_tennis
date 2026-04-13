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

        env_spacing = self.cfg.viewer.get("env_spacing", 2.5)
        scene_cfg = MJSceneCfg(num_envs=self.cfg.num_envs, env_spacing=env_spacing)

        scene_cfg.terrain = TerrainEntityCfg(
            terrain_type="plane",
            env_spacing=env_spacing,
            num_envs=self.cfg.num_envs,
        )

        from active_adaptation.assets import get_robot_cfg

        scene_cfg.entities["robot"] = get_robot_cfg(self.cfg.robot.name)

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

        scene_cfg.sensors = (contact_cfg,)

        mjlab_dt = self.cfg.sim.get("mjlab_physics_dt", None)
        if mjlab_dt is None:
            mjlab_dt = self.cfg.sim.get("mujoco_physics_dt", None)

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

        
    def _reset_idx(self, env_ids: torch.Tensor):
        init_root_state = self.command_manager.sample_init(env_ids)
        if init_root_state is not None and not self.robot.is_fixed_base:
            self.robot.write_root_state_to_sim(init_root_state, env_ids=env_ids)
        self.stats[env_ids] = 0.0
        if hasattr(self.scene, "reset"):
            self.scene.reset(env_ids)

    def render(self, mode: str = "human"):
        return super().render(mode)
