import torch

import humanoid_tennis
from humanoid_tennis.envs.base import _Env

class SimpleEnv(_Env):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.robot = self.scene["robot"]

    def setup_scene(self):
        if humanoid_tennis.get_backend() != "mjlab":
            raise NotImplementedError(
                f"Unsupported backend: {humanoid_tennis.get_backend()}"
            )

        from mjlab.scene import SceneCfg as MJSceneCfg
        from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
        from mjlab.terrains import TerrainEntityCfg
        import mjlab.terrains as terrain_gen
        from mjlab.sensor import ContactMatch, ContactSensorCfg
        from mjlab.sim import MujocoCfg, SimulationCfg
        from mjlab.scene import Scene
        from mjlab.sim.sim import Simulation
        from mjlab.utils import spec_config as spec_cfg

        mjlab_dt = self.cfg.sim.get("mjlab_physics_dt", None)
        if mjlab_dt is None:
            mjlab_dt = self.cfg.sim.get("mujoco_physics_dt", None)
        decimation_est = max(1, int(round(float(self.cfg.sim.step_dt) / float(mjlab_dt))))

        env_spacing = float(self.cfg.sim.get("env_spacing", self.cfg.viewer.get("env_spacing", 2.5)))
        print(
            "[INFO] Scene spacing: "
            f"physics_env_spacing={env_spacing:.3f} "
            f"viewer_env_spacing={float(self.cfg.viewer.get('env_spacing', env_spacing)):.3f}"
        )
        scene_cfg = MJSceneCfg(num_envs=self.cfg.num_envs, env_spacing=env_spacing)

        scene_cfg.terrain = TerrainEntityCfg(
            terrain_type="plane",
            env_spacing=env_spacing,
            num_envs=self.cfg.num_envs,
        )

        from humanoid_tennis.assets import (
            TERRAIN_BALL_BOUNCE_FRICTION,
            TERRAIN_BALL_BOUNCE_SOLREF,
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
                # Court ground stays visual-only in highlevel tennis.
                enable_racket_court_collision = False
                scene_cfg.entities["tennis_court"] = get_tennis_court_cfg(
                    texture=court_texture,
                    net_height=net_height,
                    net_collision_half_thickness=net_collision_half_thickness,
                    enable_racket_court_collision=enable_racket_court_collision,
                )
            if bool(tennis_cfg.get("add_ball", False)):
                scene_cfg.entities["tennis_ball"] = get_tennis_ball_cfg()
        # Add ball-specific collision bit (16) to terrain conaffinity while
        # preserving existing robot-foot collisions on bit 1. In tennis+ball
        # tasks, also apply the tuned bounce contact params on terrain.
        terrain_collision_kwargs = {}
        if tennis_cfg is not None and bool(tennis_cfg.get("add_ball", False)):
            terrain_collision_kwargs = {
                "friction": TERRAIN_BALL_BOUNCE_FRICTION,
                "solref": TERRAIN_BALL_BOUNCE_SOLREF,
            }
        scene_cfg.terrain.collisions = (
            spec_cfg.CollisionCfg(
                geom_names_expr=(".*",),
                contype=1,
                conaffinity=17,
                condim=3,
                disable_other_geoms=False,
                **terrain_collision_kwargs,
            ),
        )

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
                secondary=ContactMatch(mode="body", pattern="terrain"),
                fields=("found", "force"),
                reduce="maxforce",
                global_frame=False,
                num_slots=1,
                history_length=decimation_est,
                debug=False,
            )
            sensors.extend([racket_ball_sensor, ball_net_sensor, ball_court_sensor])
            
            # Contact sensor to detect racket contacts against robot body geoms.
            racket_body_sensor = ContactSensorCfg(
                name="racket_body_contact",
                primary=ContactMatch(mode="geom", pattern="tennis_racket_collision", entity="robot"),
                secondary=ContactMatch(
                    mode="geom",
                    pattern=".*_collision",
                    entity="robot",
                    # Monitor racket self-collision against robot body, excluding
                    # the racket geom itself, mounting hand, and left fake hand/wrist geoms.
                    exclude=(
                        "tennis_racket_collision",
                        "right_hand_collision",
                        "left_hand_collision",
                        "left_wrist_collision",
                    ),
                ),
                fields=("found",),
                reduce="maxforce",
                secondary_policy="any",
                global_frame=False,
                num_slots=1,
                history_length=decimation_est,
                debug=False,
            )
            sensors.append(racket_body_sensor)

        scene_cfg.sensors = tuple(sensors)

        is_tennis_ball_task = tennis_cfg is not None and bool(tennis_cfg.get("add_ball", False))
        nconmax = int(self.cfg.sim.get("nconmax", 600))
        njmax = int(self.cfg.sim.get("njmax", 2000))
        mujoco_iterations = int(self.cfg.sim.get("mujoco_iterations", 20))
        mujoco_ls_iterations = int(self.cfg.sim.get("mujoco_ls_iterations", 40))
        # Tennis scenes (ball/racket/net) require more CCD sweeps than generic tasks.
        default_ccd_iterations = 128 if is_tennis_ball_task else 50
        mujoco_ccd_iterations = int(self.cfg.sim.get("mujoco_ccd_iterations", default_ccd_iterations))
        mujoco_multiccd = bool(self.cfg.sim.get("mujoco_multiccd", False))
        # Guard against OOM in large-batch training (e.g., 2k~4k envs/GPU):
        # collision buffers in mujoco_warp scale with num_envs * nconmax *
        # (10 + 2*ccd_iterations). Keep these bounded for large jobs.
        if int(self.cfg.num_envs) >= 4096:
            capped_nconmax = min(nconmax, int(self.cfg.sim.get("mujoco_nconmax_cap", 192)))
            capped_njmax = min(njmax, int(self.cfg.sim.get("mujoco_njmax_cap", 900)))
            ccd_cap_default = 128 if is_tennis_ball_task else 96
            capped_ccd = min(mujoco_ccd_iterations, int(self.cfg.sim.get("mujoco_ccd_iterations_cap", ccd_cap_default)))
            if is_tennis_ball_task:
                capped_ccd = max(capped_ccd, int(self.cfg.sim.get("mujoco_ccd_iterations_floor", 128)))

            solver_budget = int(self.cfg.sim.get("mujoco_solver_budget", 160_000_000))
            solver_denom = max(1, 10 + 2 * capped_ccd)
            budget_nconmax = max(1, int(solver_budget // (int(self.cfg.num_envs) * solver_denom)))
            if capped_nconmax > budget_nconmax:
                print(
                    "[WARN] solver budget guard: reducing nconmax "
                    f"{capped_nconmax}->{budget_nconmax} "
                    f"(budget={solver_budget}, num_envs={int(self.cfg.num_envs)}, ccd={capped_ccd})."
                )
                capped_nconmax = budget_nconmax

            if (
                capped_nconmax != nconmax
                or capped_njmax != njmax
                or capped_ccd != mujoco_ccd_iterations
            ):
                print(
                    "[WARN] task.num_envs is large; capping mujoco buffers for stability: "
                    f"nconmax {nconmax}->{capped_nconmax}, "
                    f"njmax {njmax}->{capped_njmax}, "
                    f"ccd_iterations {mujoco_ccd_iterations}->{capped_ccd}."
                )
            nconmax = capped_nconmax
            njmax = capped_njmax
            mujoco_ccd_iterations = capped_ccd
            if mujoco_multiccd:
                print("[WARN] task.num_envs is large; forcing mujoco_multiccd=false to avoid GPU OOM.")
            mujoco_multiccd = False
        solver_load_index = int(self.cfg.num_envs) * int(nconmax) * (10 + 2 * int(mujoco_ccd_iterations))
        print(
            "[INFO] Mujoco solver profile: "
            f"num_envs={int(self.cfg.num_envs)} nconmax={int(nconmax)} njmax={int(njmax)} "
            f"iters={int(mujoco_iterations)}/{int(mujoco_ls_iterations)} "
            f"ccd={int(mujoco_ccd_iterations)} multiccd={bool(mujoco_multiccd)} "
            f"solver_load_index={solver_load_index}"
        )
        self.sim_cfg = sim_cfg = SimulationCfg(
            nconmax=nconmax,
            njmax=njmax,
            mujoco=MujocoCfg(
                timestep=mjlab_dt,
                iterations=mujoco_iterations,
                ls_iterations=mujoco_ls_iterations,
                ccd_iterations=mujoco_ccd_iterations,
                multiccd=mujoco_multiccd,
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
        if hasattr(self.scene, "reset"):
            self.scene.reset(env_ids)
        self._align_tennis_court_to_env_origins(env_ids)
        init_root_state = self.command_manager.sample_init(env_ids)
        if init_root_state is not None and not self.robot.is_fixed_base:
            self.robot.write_root_state_to_sim(init_root_state, env_ids=env_ids)
        self.stats[env_ids] = 0.0

    def render(self, mode: str = "human"):
        return super().render(mode)
