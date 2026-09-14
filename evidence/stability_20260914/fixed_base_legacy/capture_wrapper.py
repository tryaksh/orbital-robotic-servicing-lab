"""Capture the published Isaac mission without altering its controllers or sensors.

This read-only source wrapper adds pose observations and a scene export. The
result is a fresh simulation episode, not a replay of the older seed-6070 video.
All outputs stay in the portfolio workspace; the original project is untouched.
Run with C:/isaac-sim/python.bat from the portfolio repository.
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path('D:/portfolio/.lab-work/orbital-improvement/source')
sys.path.insert(0, str(ROOT / 'src'))
OUT = Path('D:/portfolio/.lab-work/orbital-improvement/no-rail-original')
OUT.mkdir(parents=True, exist_ok=True)
source_path = ROOT / 'scripts/run_workflow_demo.py'
source = source_path.read_text(encoding='utf-8')
job_path = Path('D:/6axis-space-robotics') / 'artifacts/mission_final_seed6070/jobs/65461e26-e57f-4374-af19-913c3312d9c9/job.json'
job = json.loads(job_path.read_text())
argv = job['provenance']['command_argv'][2:]
argv = [value.replace("D:\\6axis-space-robotics", str(ROOT)).replace("D:/6axis-space-robotics", str(ROOT)) for value in argv]
for flag, value in [('--seed', '6070'), ('--num_envs', '1'), ('--steps', '1900')]:
    argv[argv.index(flag) + 1] = value
if 'none' == 'none':
    argv.remove('--robot_rail_on_relocation')

for flag, value in [('--video_dir', OUT/'video'), ('--handoff_trace', OUT/'handoff_trace.npz'), ('--report', OUT/'workflow_report.json')]:
    argv[argv.index(flag) + 1] = str(value)
sys.argv = [str(source_path), *argv]

hook = '''
        # Portfolio-only readback: no actions, conditions or sensor settings change.
        import omni.usd
        from pxr import UsdGeom, Usd
        _portfolio_stage = omni.usd.get_context().get_stage()
        _portfolio_out = Path("D:/portfolio/.lab-work/orbital-improvement/no-rail-original")
        _portfolio_bodies = []
        for _name, _asset in task.scene.articulations.items():
            _paths = _asset.root_physx_view.link_paths[0]
            for _idx, _path in enumerate(_paths):
                _portfolio_bodies.append((str(_path), _asset, _idx))
        for _name, _asset in task.scene.rigid_objects.items():
            _portfolio_bodies.append((str(_asset.root_physx_view.prim_paths[0]), _asset, 0))
        _portfolio_paths = [row[0] for row in _portfolio_bodies]
        _portfolio_rows = []
        _portfolio_joint_targets = []
        _portfolio_actions = []
        _portfolio_aux = []
        # Rack pawls / robot-side lock visual pieces may be USD-driven rather than rigid bodies.
        _portfolio_aux_prims = []
        for _prim in _portfolio_stage.Traverse():
            _path = str(_prim.GetPath())
            if ('Pawl' in _path or 'pawl' in _path or 'ServiceLatch' in _path) and _prim.IsA(UsdGeom.Xform):
                if _path not in _portfolio_paths:
                    _portfolio_aux_prims.append(_prim)
        def _portfolio_sample():
            _portfolio_joint_targets.append(task.scene["robot"].data.joint_pos_target[0, driver.arm_joint_ids].cpu().numpy().copy())
            _portfolio_actions.append(driver.actions[0].cpu().numpy().copy())
            _portfolio_rows.append(np.stack([_asset.data.body_link_pose_w[0, _idx].cpu().numpy().copy() for _, _asset, _idx in _portfolio_bodies]))
            _portfolio_aux.append(np.stack([np.asarray(UsdGeom.Xformable(_prim).GetLocalTransformation()) for _prim in _portfolio_aux_prims]) if _portfolio_aux_prims else np.empty((0,4,4)))
        _portfolio_sample()
        _portfolio_stage.Export(str(_portfolio_out / 'scene.usda'))
        print('[PORTFOLIO] Capturing', len(_portfolio_bodies), 'body paths and', len(_portfolio_aux_prims), 'visual paths', flush=True)
'''
needle = '        collecting = args.episodes > 0\n'
assert source.count(needle) == 1
source = source.replace(needle, hook + '\n' + needle)
needle = '            _, _, terminated, truncated, _ = env.step(driver.actions)\n'
assert source.count(needle) == 1
source = source.replace(needle, needle + '            _portfolio_sample()\n')
needle = '        # Multi-environment diagnostics are already a fixed cohort:'
assert source.count(needle) == 1
source = source.replace(needle, '''        np.savez_compressed(_portfolio_out / "body_poses.npz", poses=np.stack(_portfolio_rows), paths=np.array(_portfolio_paths), aux_matrices=np.stack(_portfolio_aux), aux_paths=np.array([str(p.GetPath()) for p in _portfolio_aux_prims]), fps=np.array(30), arm_joint_targets=np.stack(_portfolio_joint_targets), controller_actions=np.stack(_portfolio_actions))
        print('[PORTFOLIO] Saved', len(_portfolio_rows), 'frames', flush=True)

''' + needle)
(OUT / 'capture_provenance.json').write_text(json.dumps({
    'description': 'Fresh live camera-driven mission, published controllers/configuration plus read-only pose capture.',
    'original_source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
    'instrumented_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
    'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'source_commit': 'a2e017b79e18d9d58106aaf0dcc943471b468920',
    'seed': 6070, 'original_command': argv,
    'scope': 'One simulated episode; no new reliability rate. Visual replay changes presentation only.'
}, indent=2))
exec(compile(source, str(source_path), 'exec'), {'__name__': '__main__', '__file__': str(source_path)})
