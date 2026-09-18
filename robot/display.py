from pathlib import Path

import mujoco
import numpy as np


class RobotDisplay:
    """Display the supplied URDF and copy LampController values into it."""

    JOINTS = {
        "base_yaw_joint": "base_yaw",
        "shoulder_pitch_joint": "shoulder_pitch",
        "elbow_pitch_joint": "elbow_pitch",
        "neck_yaw_joint": "neck_yaw",
        "head_pitch_joint": "head_pitch",
    }

    def __init__(self, width=640, height=480):
        urdf_path = Path(__file__).with_name("dummy_lamp_5dof.urdf")
        self.model = mujoco.MjModel.from_xml_path(str(urdf_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, width=width, height=height)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = (0.0, 0.0, 0.32)
        self.camera.distance = 1.25
        self.camera.azimuth = 135
        self.camera.elevation = -15

        self.joint_addresses = {
            joint_name: self.model.joint(joint_name).qposadr[0]
            for joint_name in self.JOINTS
        }
        # The light material has a unique warm color in the supplied URDF.
        self.light_geoms = [
            index
            for index, color in enumerate(self.model.geom_rgba)
            if np.allclose(color[:3], (1.0, 0.95, 0.76), atol=0.02)
        ]

    def render(self, lamp):
        for joint_name, lamp_attribute in self.JOINTS.items():
            address = self.joint_addresses[joint_name]
            target = getattr(lamp, lamp_attribute)
            # ArmTracker already smooths the target; filtering again adds lag.
            self.data.qpos[address] = target

        light_color = (
            (1.0, 0.82, 0.18, 1.0)
            if lamp.light_on
            else (0.22, 0.22, 0.20, 1.0)
        )
        for geom_id in self.light_geoms:
            self.model.geom_rgba[geom_id] = light_color

        mujoco.mj_forward(self.model, self.data)

        self.renderer.update_scene(self.data, self.camera)
        return self.renderer.render()

    def close(self):
        self.renderer.close()
