from scipy.spatial.transform import Rotation as sRot, Slerp
import numpy as np

def lerp(x, xp, fp):
    """
    线性插值：对多维数据的每个维度分别插值
    x: 目标时间戳数组 (M,)
    xp: 原始时间戳数组 (T,)
    fp: 原始数据 (T, D)
    返回 (M, D)
    """
    return np.stack([np.interp(x, xp, fp[:, i]) for i in range(fp.shape[1])], axis=-1)

def slerp_quat(x, xp, fp):
    """
    支持对 (T,4) 或 (T,B,4) 形状的四元数数据做球面线性插值。
    x: 目标时间点 (M,)
    xp: 原始时间点 (T,)
    fp: 原始四元数，最后一维为 4
    返回形状：(M,4) 或 (M,B,4)
    """
    arr = np.asarray(fp)
    if arr.ndim == 2 and arr.shape[1] == 4:
        # 单个四元数序列
        rot = Slerp(xp, sRot.from_quat(arr, scalar_first=True))
        return rot(x).as_quat(scalar_first=True)
    elif arr.ndim >= 3 and arr.shape[-1] == 4:
        # 多个并行四元数序列
        T = arr.shape[0]
        # 展平中间所有维度
        flat = arr.reshape(T, -1, 4)
        M = len(x)
        np_out = np.zeros((M, flat.shape[1], 4))
        for i in range(flat.shape[1]):
            rot_i = Slerp(xp, sRot.from_quat(flat[:, i, :], scalar_first=True))
            np_out[:, i, :] = rot_i(x).as_quat(scalar_first=True)
        # 恢复原始中间维度
        out_shape = (M,) + arr.shape[1:]
        return np_out.reshape(out_shape)
    else:
        raise ValueError(f"Unexpected quaternion array shape: {arr.shape}")

def interpolate(motion, target_fps: int = 50):
    """
    将 motion 数据重采样至 target_fps
    支持 qpos, qvel, xpos, xquat, cvel
    """
    if motion.get("fps", 0) != target_fps:
        breakpoint()
        T = motion["qpos"].shape[0]
        end_t = T / motion["fps"]
        xp = np.arange(0, end_t, 1 / motion["fps"])
        x = np.arange(0, end_t, 1 / target_fps)
        if x[-1] > xp[-1]:
            x = x[:-1]
        # 插值 qpos
        motion["qpos"] = lerp(x, xp, motion["qpos"])
        # 重计算 qvel
        # body 世界坐标位置
        T2 = motion["xpos"].shape[0]
        motion["xpos"] = lerp(x, xp, motion["xpos"].reshape(T2, -1)).reshape(len(x), *motion["xpos"].shape[1:])
        # # body 世界四元数
        # motion["xquat"] = slerp_quat(x, xp, motion["xquat"].reshape(T2, -1, 4)).reshape(len(x), *motion["xquat"].shape[1:])
        # # 空间速度
        # motion["cvel"] = lerp(x, xp, motion["cvel"].reshape(T2, -1)).reshape(len(x), *motion["cvel"].shape[1:])
        motion["fps"] = target_fps
    dq = np.diff(motion["qpos"], axis=0) * target_fps
    motion["qvel"] = np.concatenate([dq, dq[-1:]], axis=0)
    return motion

def rotate_to_body(root_quat, vecs):
    """
    将世界系向量旋转到以 root 四元数定义的本体系
    root_quat: (T,4) scalar-last
    vecs: (T, N, 3)
    返回 (T, N, 3)
    """
    r = sRot.from_quat(root_quat, scalar_first=True)
    inv = r.inv().as_matrix()  # (T,3,3)
    return np.einsum('tij,tnj->tni', inv, vecs)

def finite_diff_vel(x: np.ndarray, fps: float) -> np.ndarray:
    """
    Central difference velocity for a time sequence.
    x: (T, ...) array
    fps: frames per second
    """
    arr = np.asarray(x)
    vel = np.zeros_like(arr, dtype=arr.dtype)
    if fps <= 0 or arr.shape[0] < 2:
        return vel
    vel[1:-1] = (arr[2:] - arr[:-2]) * (fps / 2.0)
    vel[0] = (arr[1] - arr[0]) * fps
    vel[-1] = (arr[-1] - arr[-2]) * fps
    return vel

from typing import Sequence, Tuple, List, Any, Union
import numpy as np

def select_in_order(
    original: Union[Sequence[Any], np.ndarray],
    whitelist: Sequence[Any],
    return_missing: bool = False,
) -> Union[
    Tuple[List[Any], List[int]],
    Tuple[np.ndarray, np.ndarray],
    Tuple[List[Any], List[int], List[Any]],
    Tuple[np.ndarray, np.ndarray, List[Any]]
]:
    """
    按白名单顺序从 original 中筛选元素。
    - original: 一维列表/元组/np.ndarray（元素通常是字符串）
    - whitelist: 希望的顺序（可以有缺失）
    - return_missing: 为 True 时，额外返回 whitelist 中未在 original 出现的项

    返回:
      selected: 筛选后的元素（顺序与 whitelist 一致）
      idx     : 这些元素在 original 中的索引（可用于原数据切片）
      missing : （可选）whitelist 中缺失的元素列表
    """
    # 统一成 Python list 做映射
    if isinstance(original, np.ndarray):
        if original.ndim != 1:
            raise ValueError("original 必须是一维序列/数组")
        original_list = original.tolist()
        return_np = True
    else:
        original_list = list(original)
        return_np = False

    # 建立值 -> 首次出现索引 的映射（如果 original 里有重复，取第一个）
    index_map = {}
    for i, v in enumerate(original_list):
        if v not in index_map:
            index_map[v] = i

    # 按 whitelist 顺序选取存在于 original 的元素
    selected = [x for x in whitelist if x in index_map]
    idx = [index_map[x] for x in selected]
    missing = [x for x in whitelist if x not in index_map] if return_missing else None

    # 维持返回类型与 original 一致
    if return_np:
        selected = np.array(selected, dtype=object if original.dtype == object else original.dtype)
        idx = np.array(idx, dtype=int)
        return (selected, idx, missing) if return_missing else (selected, idx)
    else:
        return (selected, idx, missing) if return_missing else (selected, idx)

import numpy as np
from typing import Union
from scipy.spatial.transform import Rotation as sRot

def angvel_from_rot(rot: Union[np.ndarray, sRot], fps: float, quat_order: str = "xyzw") -> np.ndarray:
    """
    使用四元数导数稳健估计角速度（世界系）。
    公式：omega_world = 2 * ( qdot ⊗ conj(q) ).vec

    参数
    ----
    rot : np.ndarray[T,4] (四元数) 或 np.ndarray[T,3,3] (旋转矩阵) 或 scipy Rotation
        若为四元数数组，默认顺序为 xyzw；可通过 quat_order 指定为 "wxyz"。
    fps : float
        采样频率（Hz），用于差分求导。
    quat_order : {"xyzw","wxyz"}
        指定四元数输入顺序，默认 "xyzw"。

    返回
    ----
    np.ndarray[T,3] : 世界系角速度（rad/s）。
    """
    if fps <= 0:
        raise ValueError("fps must be positive.")
    # 取出四元数（xyzw）
    if isinstance(rot, sRot):
        quat_xyzw = rot.as_quat().astype(np.float64)  # (T,4), xyzw
    else:
        arr = np.asarray(rot)
        if arr.ndim == 2 and arr.shape[1] == 4:
            quat_xyzw = arr.astype(np.float64)  # (T,4)
            if quat_order.lower() == "wxyz":
                # 转为 xyzw
                quat_xyzw = np.concatenate([quat_xyzw[:, 1:], quat_xyzw[:, :1]], axis=-1)
            elif quat_order.lower() != "xyzw":
                raise ValueError("quat_order must be 'xyzw' or 'wxyz'.")
        elif arr.ndim == 3 and arr.shape[1:] == (3, 3):
            quat_xyzw = sRot.from_matrix(arr).as_quat().astype(np.float64)  # (T,4), xyzw
        else:
            raise ValueError("rot must be Rotation, (T,4) quats, or (T,3,3) matrices.")

    T = quat_xyzw.shape[0]
    if T == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if T == 1:
        return np.zeros((1, 3), dtype=np.float32)

    # 转为 wxyz 并归一化
    q_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=-1)  # (T,4)
    q_wxyz /= np.linalg.norm(q_wxyz, axis=1, keepdims=True).clip(min=1e-12)

    # 解缠：强制相邻点积 >= 0，避免 q/-q 翻转
    dots = np.sum(q_wxyz[1:] * q_wxyz[:-1], axis=1)
    flip_idx = np.where(dots < 0)[0] + 1  # 注意：布尔索引会产生拷贝，这里用显式索引
    if flip_idx.size > 0:
        q_wxyz[flip_idx] *= -1.0

    # 中心差分/端点单边差分，得到 qdot（单位：1/s）
    qdot = np.zeros_like(q_wxyz)
    qdot[1:-1] = (q_wxyz[2:] - q_wxyz[:-2]) * (fps / 2.0)
    qdot[0]    = (q_wxyz[1]  - q_wxyz[0])  * fps
    qdot[-1]   = (q_wxyz[-1] - q_wxyz[-2]) * fps

    # Hamilton 乘法（wxyz）
    def qmul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return np.stack([
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw
        ], axis=-1)

    # 共轭（单位四元数的逆）
    q_conj = q_wxyz.copy()
    q_conj[:, 1:] *= -1.0

    # omega_world = 2 * (qdot ⊗ q_conj).vec
    omega_quat = qmul_wxyz(qdot, q_conj) * 2.0
    omega_world = omega_quat[:, 1:]  # 取向量部

    return omega_world.astype(np.float32)

def angvel_from_quat_wxyz_batch(quat_wxyz: np.ndarray, fps: float) -> np.ndarray:
    """
    Compute angular velocity for batched quaternions in wxyz order.
    quat_wxyz: (T, B, 4)
    return: (T, B, 3)
    """
    arr = np.asarray(quat_wxyz)
    if arr.ndim != 3 or arr.shape[-1] != 4:
        raise ValueError(f"quat_wxyz must be (T,B,4), got {arr.shape}")
    T, B, _ = arr.shape
    out = np.zeros((T, B, 3), dtype=np.float32)
    for i in range(B):
        out[:, i] = angvel_from_rot(arr[:, i], fps=fps, quat_order="wxyz")
    return out
