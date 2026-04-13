import os
import sys
import time
import yaml
from multiprocessing import Process, Value
from typing import Dict, Optional

import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, MotorMode
from common.remote_controller import RemoteController, KeyMap
from common.utils import DictToClass, Timer
from common.joint_mapper import create_isaac_to_real_mapper

from policy import Policy, TrackingPolicyRaw
from pathlib import Path
from paths import REAL_G1_ROOT

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

def get_config(policy_cfg_path: str) -> DictToClass:
    policy_cfg_path = Path(policy_cfg_path)
    if not policy_cfg_path.is_absolute():
        policy_cfg_path = REAL_G1_ROOT / policy_cfg_path
    with open(str(policy_cfg_path), 'r') as f:
        policy_cfg = DictToClass(yaml.load(f, Loader=yaml.FullLoader))
    return policy_cfg

class Controller:
    def __init__(self, args, ctrl_cfg):
        self.args = args
        self.config = ctrl_cfg
        self.remote_controller = RemoteController()
        self.control_dt = 1.0 / self.config.control_freq

        self.isaac_to_real_mapper_state = create_isaac_to_real_mapper(
            self.config.isaac_joint_names_state,
            self.config.real_joint_names
        )
        m_info = self.isaac_to_real_mapper_state.get_mapping_info()
        print(f"[Controller] State mapping: {m_info['mapped_joints']}/{m_info['from_space_size']} mapped")

        self.dof_size_real = len(self.config.real_joint_names)

        self.qj_real = np.zeros(self.dof_size_real, dtype=np.float32)
        self.dqj_real = np.zeros(self.dof_size_real, dtype=np.float32)
        self.tau_real = np.zeros(self.dof_size_real, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.gyro = np.zeros(3, dtype=np.float32)
        self.linacc = np.zeros(3, dtype=np.float32)

        self.qj_isaac = None
        self.dqj_isaac = None
        self.tau_isaac = None

        self.default_qpos_real = np.array(self.config.default_qpos_real, dtype=np.float32)
        self.init_qpos_real    = np.array(self.config.init_qpos_real, dtype=np.float32)
        self.kps_real          = np.array(self.config.kps_real, dtype=np.float32)
        self.kds_real          = np.array(self.config.kds_real, dtype=np.float32)
        self.dof_size_real = len(self.default_qpos_real)

        self.counter = 0
        self.policy_step = 0
        self.is_alive = True

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0

        self.lowcmd_publisher_ = ChannelPublisher(self.config.lowcmd_topic, LowCmdHG)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber(self.config.lowstate_topic, LowStateHG)
        self.lowstate_subscriber.Init()

        self.loop_count = Value('i', 0)
        self.p_loop_rate = Process(target=self.count_loop_rate, args=(self.loop_count,), daemon=True)

        self.wait_for_low_state()
        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

        self.policies = {
            "tracking": TrackingPolicyRaw("tracking", get_config("config/tracking.yaml"), self),
        }
        self.current_policy: Optional[Policy] = None
        self.pending_policy: Optional[Policy] = None

        self._prev_buttons = None
        self.btn_rise = None
        self.btn_fall = None

    def count_loop_rate(self, loop_count):
        count_loop_timer = Timer(1.0)
        while self.is_alive:
            if loop_count.value < self.config.control_freq - 2 or loop_count.value > self.config.control_freq + 2:
                print(f'[Warning] Loop rate: {loop_count.value} Hz')
            loop_count.value = 0
            count_loop_timer.sleep()

    def _consume_low_state(self, msg: LowStateHG) -> bool:
        if msg is None or not hasattr(msg, "motor_state"):
            return False

        self.low_state = msg
        self.qj_real[:] = np.array([msg.motor_state[i].q for i in range(self.dof_size_real)], dtype=np.float32)
        self.dqj_real[:] = np.array([msg.motor_state[i].dq for i in range(self.dof_size_real)], dtype=np.float32)
        self.tau_real[:] = np.array([msg.motor_state[i].tau_est for i in range(self.dof_size_real)], dtype=np.float32)
        self.quat[:] = np.array(msg.imu_state.quaternion, dtype=np.float32)
        self.gyro[:] = np.array(msg.imu_state.gyroscope, dtype=np.float32)
        self.linacc[:] = np.array(msg.imu_state.accelerometer, dtype=np.float32)

        if self.args.sim2sim:
            self.remote_controller.set_sim2sim(msg.wireless_remote)
        elif self.args.real:
            self.remote_controller.set(msg.wireless_remote)

        self.mode_machine_ = msg.mode_machine
        return True

    def send_cmd(self, cmd):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            msg = self.lowstate_subscriber.Read()
            self._consume_low_state(msg)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            self.process_state()
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

    def move_to_default_qpos(self):
        print("Moving to init pos....")
        total_time = 2.0
        num_step = int(total_time / self.control_dt)

        init_dof_pos = np.zeros(self.dof_size_real, dtype=np.float32)
        for i in range(self.dof_size_real):
            init_dof_pos[i] = self.low_state.motor_state[i].q

        for t in range(num_step):
            alpha = t / num_step
            for i in range(self.dof_size_real):
                target_pos = self.init_qpos_real[i]
                self.low_cmd.motor_cmd[i].q  = init_dof_pos[i] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[i].qd = 0
                self.low_cmd.motor_cmd[i].kp = self.kps_real[i]
                self.low_cmd.motor_cmd[i].kd = self.kds_real[i]
                self.low_cmd.motor_cmd[i].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

    def default_qpos_state(self):
        initial_policy: Optional[Policy] = None

        print("Press A to tracking policy...")

        while True:
            self.process_state()
            
            for i in range(self.dof_size_real):
                self.low_cmd.motor_cmd[i].q  = self.init_qpos_real[i]
                self.low_cmd.motor_cmd[i].qd = 0
                self.low_cmd.motor_cmd[i].kp = self.kps_real[i]
                self.low_cmd.motor_cmd[i].kd = self.kds_real[i]
                self.low_cmd.motor_cmd[i].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

            if self.btn_rise[KeyMap.select] == 1:
                raise KeyboardInterrupt

            if self.btn_rise[KeyMap.A]:
                initial_policy = self.policies["tracking"]
                print("Initial policy: tracking")
                break

        self.current_policy = initial_policy
        if hasattr(self.current_policy, "kps_real") and hasattr(self.current_policy, "kds_real"):
            self.kps_real[:] = self.current_policy.kps_real
            self.kds_real[:] = self.current_policy.kds_real
            print(f"[Controller] Updated gains to policy defaults.")
        self.current_policy.fade_in()
        self.low_cmd.reserve[0] = 1
        self.send_cmd(self.low_cmd)

    def process_state(self):
        msg = self.lowstate_subscriber.Read(0.0)
        self._consume_low_state(msg)

        now = np.array(self.remote_controller.button, dtype=np.int8)
        if self._prev_buttons is None or len(self._prev_buttons) != len(now):
            self._prev_buttons = now.copy()
            self.btn_rise = np.zeros_like(now, dtype=bool)
            self.btn_fall = np.zeros_like(now, dtype=bool)
        else:
            self.btn_rise = (self._prev_buttons == 0) & (now == 1)
            self.btn_fall = (self._prev_buttons == 1) & (now == 0)
            self._prev_buttons = now

        self.qj_isaac = self.isaac_to_real_mapper_state.map_state_to_from(self.qj_real)
        self.dqj_isaac = self.isaac_to_real_mapper_state.map_state_to_from(self.dqj_real)
        self.tau_isaac = self.isaac_to_real_mapper_state.map_state_to_from(self.tau_real)

    def _apply_action_real(self, action_real_delta: np.ndarray):
        if action_real_delta is None or not np.all(np.isfinite(action_real_delta)):
            print("[Controller] action invalid; hold init PD")
            raise KeyboardInterrupt
        else:
            desired = self.default_qpos_real + action_real_delta
            target = desired

        for i in range(self.dof_size_real):
            self.low_cmd.motor_cmd[i].q  = float(target[i])
            self.low_cmd.motor_cmd[i].qd = 0.0
            self.low_cmd.motor_cmd[i].kp = float(self.kps_real[i])
            self.low_cmd.motor_cmd[i].kd = float(self.kds_real[i])
            self.low_cmd.motor_cmd[i].tau = 0.0

    def run(self):
        print("Running high level...")
        self.p_loop_rate.start()
        timer = Timer(self.control_dt)
        loop_count = self.loop_count

        try:
            while True:
                self.process_state()

                if self.btn_rise[KeyMap.select] == 1:
                    break

                self.current_policy.update_obs()
                action_real = self.current_policy.compute_action()
                self._apply_action_real(action_real)
                self.current_policy.post_step()

                self.send_cmd(self.low_cmd)
                loop_count.value += 1
                self.policy_step += 1
                timer.sleep()
        finally:
            pass

    def close(self):
        print("Closing...")
        self.is_alive = False
        if self.p_loop_rate is not None and self.p_loop_rate.is_alive():
            self.p_loop_rate.terminate()
        sys.exit(0)

import traceback

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, default=None)
    parser.add_argument("--sim2sim", action='store_true')
    parser.add_argument("--real", action='store_true')
    args = parser.parse_args()
    assert args.sim2sim ^ args.real, "Please specify either sim2sim or real."

    ChannelFactoryInitialize(0, args.net)

    controller = Controller(args, get_config("config/controller.yaml"))

    controller.zero_torque_state()
    controller.move_to_default_qpos()
    try:
        controller.default_qpos_state()
        controller.run()
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    except Exception as e:
        print(f"An exception occurred: {e}")
        traceback.print_exc()
    finally:
        create_damping_cmd(controller.low_cmd)
        controller.send_cmd(controller.low_cmd)
        controller.close()
