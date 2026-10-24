"""Just open the MuJoCo scene in the passive viewer — no policy, no training.

Run with:   mjpython view_scene.py
"""

import time
import mujoco
import mujoco.viewer as mj_viewer

m = mujoco.MjModel.from_xml_path("robot_pickplace.xml")
d = mujoco.MjData(m)

with mj_viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(m.opt.timestep)
