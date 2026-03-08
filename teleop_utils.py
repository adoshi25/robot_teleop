
import mujoco
import numpy as np
import pyroki as pk
import jax
import jax.numpy as jnp
import jaxlie
import jaxls
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp


def _slerp_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical linear interpolation between quaternions in wxyz format."""
    # scipy uses xyzw: [x, y, z, w]
    xyzw0 = np.array([q0[1], q0[2], q0[3], q0[0]])
    xyzw1 = np.array([q1[1], q1[2], q1[3], q1[0]])
    r0 = Rotation.from_quat(xyzw0)
    r1 = Rotation.from_quat(xyzw1)
    slerp = Slerp([0, 1], Rotation.concatenate([r0, r1]))
    r = slerp(alpha)
    xyzw = r.as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def _get_robot_base_poses_from_model(model):
    def _pose(body_name):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return {
            "pos": np.array(model.body_pos[bid], dtype=float),
            "quat": np.array(model.body_quat[bid], dtype=float),
        }

    return {
        "left": _pose("left_robot_base"),
        "right": _pose("right_robot_base"),
    }


# Max IK iterations for real-time: warm start converges quickly; cold needs more.
_IK_MAX_ITERATIONS = 12

def _solve_ik_core_jax(
    robot,
    target_link_index: jnp.ndarray,
    target_position: jnp.ndarray,
    target_wxyz: jnp.ndarray,
) -> jnp.ndarray:
    """JIT-compiled IK core (no warm start). Returns (8,) joint config."""
    joint_var = robot.joint_var_cls(0)
    target_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_wxyz), target_position
    )
    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot, joint_var, target_pose, target_link_index,
            pos_weight=50.0, ori_weight=10.0,
        ),
        pk.costs.limit_constraint(robot, joint_var),
    ]
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            termination=jaxls.TerminationConfig(max_iterations=_IK_MAX_ITERATIONS),
        )
    )
    return sol[joint_var]


def _solve_ik_core_jax_warm(
    robot,
    target_link_index: jnp.ndarray,
    target_position: jnp.ndarray,
    target_wxyz: jnp.ndarray,
    initial_q: jnp.ndarray,
) -> jnp.ndarray:
    """JIT-compiled IK with warm start. Returns (8,) joint config."""
    joint_var = robot.joint_var_cls(0)
    target_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_wxyz), target_position
    )
    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot, joint_var, target_pose, target_link_index,
            pos_weight=50.0, ori_weight=10.0,
        ),
        pk.costs.limit_constraint(robot, joint_var),
    ]
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            termination=jaxls.TerminationConfig(max_iterations=_IK_MAX_ITERATIONS),
            initial_vals=jaxls.VarValues.make((joint_var.with_value(initial_q),)),
        )
    )
    return sol[joint_var]


def _solve_ik_core_balanced(
    robot,
    target_link_index: jnp.ndarray,
    target_position: jnp.ndarray,
    target_wxyz: jnp.ndarray,
) -> jnp.ndarray:
    """JIT-compiled IK with balanced weights (no warm start)."""
    joint_var = robot.joint_var_cls(0)
    target_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_wxyz), target_position
    )
    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot, joint_var, target_pose, target_link_index,
            pos_weight=50.0, ori_weight=5.0,
        ),
        pk.costs.limit_constraint(robot, joint_var),
    ]
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            termination=jaxls.TerminationConfig(max_iterations=_IK_MAX_ITERATIONS),
        )
    )
    return sol[joint_var]


def _solve_ik_core_balanced_warm(
    robot,
    target_link_index: jnp.ndarray,
    target_position: jnp.ndarray,
    target_wxyz: jnp.ndarray,
    initial_q: jnp.ndarray,
) -> jnp.ndarray:
    """JIT-compiled IK with balanced weights and warm start."""
    joint_var = robot.joint_var_cls(0)
    target_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_wxyz), target_position
    )
    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot, joint_var, target_pose, target_link_index,
            pos_weight=50.0, ori_weight=5.0,
        ),
        pk.costs.limit_constraint(robot, joint_var),
    ]
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            termination=jaxls.TerminationConfig(max_iterations=_IK_MAX_ITERATIONS),
            initial_vals=jaxls.VarValues.make((joint_var.with_value(initial_q),)),
        )
    )
    return sol[joint_var]


_solve_ik_jit = jax.jit(_solve_ik_core_jax)
_solve_ik_warm_jit = jax.jit(_solve_ik_core_jax_warm)
_solve_ik_balanced_jit = jax.jit(_solve_ik_core_balanced)
_solve_ik_balanced_warm_jit = jax.jit(_solve_ik_core_balanced_warm)
_solve_ik_batch_jit = jax.jit(jax.vmap(_solve_ik_core_jax, in_axes=(None, None, 0, None)))
_solve_ik_batch_poses_jit = jax.jit(
    jax.vmap(_solve_ik_core_jax, in_axes=(None, None, 0, 0))
)


# ori_weight value -> (cold_jit, warm_jit) lookup; pre-built for common values
_ORI_WEIGHT_DISPATCH = {
    10.0: (_solve_ik_jit, _solve_ik_warm_jit),
    5.0: (_solve_ik_balanced_jit, _solve_ik_balanced_warm_jit),
}


def _solve_ik_pyroki(
    robot,
    target_link_name,
    target_position,
    target_wxyz,
    initial_q=None,
    ori_weight=5.0,
):
    """
    Solve IK using pyroki. Returns 7-DOF joint config (arm only).
    initial_q: Optional (7,) for warm start (faster convergence when streaming).
    ori_weight: Orientation cost weight (0=position-only, 5=balanced, 10=original).
    """
    target_link_index = jnp.array(
        robot.links.names.index(target_link_name), dtype=jnp.int32
    )
    pos = jnp.asarray(target_position, dtype=jnp.float32)
    wxyz = jnp.asarray(target_wxyz, dtype=jnp.float32)

    cold_fn, warm_fn = _ORI_WEIGHT_DISPATCH.get(
        ori_weight, (_solve_ik_balanced_jit, _solve_ik_balanced_warm_jit)
    )

    if initial_q is not None:
        q0 = jnp.asarray(initial_q, dtype=jnp.float32)
        init = jnp.concatenate([q0.reshape(-1)[:7], jnp.array([0.0])])[:8]
        q = warm_fn(robot, target_link_index, pos, wxyz, init)
    else:
        q = cold_fn(robot, target_link_index, pos, wxyz)
    return np.array(q[:7])


def _solve_ik_pyroki_batch(
    robot,
    target_link_name,
    target_positions,
    target_wxyz,
    target_wxyz_per_point=None,
) -> np.ndarray:
    """
    Solve IK for multiple targets in one batched call (GPU-parallel).
    target_positions: (N, 3).
    target_wxyz: (4,) shared orientation, or ignored if target_wxyz_per_point given.
    target_wxyz_per_point: (N, 4) per-target orientation (wxyz). Overrides target_wxyz.
    Returns (N, 7).
    """
    target_link_index = jnp.array(
        robot.links.names.index(target_link_name), dtype=jnp.int32
    )
    positions = jnp.asarray(target_positions, dtype=jnp.float32)
    if target_wxyz_per_point is not None:
        wxyz = jnp.asarray(target_wxyz_per_point, dtype=jnp.float32)
        qs = _solve_ik_batch_poses_jit(robot, target_link_index, positions, wxyz)
    else:
        wxyz = jnp.asarray(target_wxyz, dtype=jnp.float32)
        qs = _solve_ik_batch_jit(robot, target_link_index, positions, wxyz)
    return np.array(qs)[:, :7]

def _spline_interpolate(waypoints, timestamps, num_samples):
    """Resample waypoints to num_samples using clamped cubic splines."""
    if len(waypoints) < 2:
        return waypoints, timestamps

    sample_times = np.linspace(timestamps[0], timestamps[-1], num_samples)
    cs = CubicSpline(timestamps, waypoints, bc_type="clamped")
    return cs(sample_times), sample_times


def _smooth_trajectory(joint_angles, sigma=1.5):
    """Gaussian-smooth a joint trajectory along the time axis."""
    if len(joint_angles) < 3:
        return joint_angles
    return gaussian_filter1d(joint_angles.copy(), sigma=sigma, axis=0, mode="nearest")
