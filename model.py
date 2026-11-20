"""Faithful 3-DOF yaw-pitch-pitch MJCF model from the original pidmujoco code.

Original source: https://github.com/vitor-souza-ime/pidmujoco (commit
055319e251a11f1042b974f8a897a6300fa2ed1d).  The only intended
physical change in this model is optional end-effector payload mass.
"""


def robot_xml(payload_mass: float = 0.1) -> str:
    if payload_mass <= 0:
        raise ValueError("Payload mass must be positive")
    # Preserve the original 0.001 kg m^2 inertia at 0.1 kg; scale inertia
    # proportionally with the added payload mass at constant dimensions.
    payload_inertia = 0.001 * payload_mass / 0.1
    return f"""
<mujoco model="pidmujoco_yaw_pitch_pitch">
    <compiler angle="degree"/>
    <option timestep="0.005" gravity="0 0 -9.81"/>
    <worldbody>
        <geom name="floor" pos="0 0 -0.5" size="1 1 0.1" type="plane" rgba="0.5 0.5 0.5 1"/>
        <body name="base" pos="0 0 0">
            <geom name="base_geom" type="cylinder" size="0.1 0.1" rgba="0.2 0.2 0.8 1"/>
            <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
            <joint name="joint1" type="hinge" axis="0 0 1" range="-180 180" damping="0.1"/>
            <body name="link1" pos="0 0 0.1">
                <geom name="link1_geom" type="capsule" fromto="0 0 0 0.3 0 0" size="0.05" rgba="0.8 0.2 0.2 1"/>
                <inertial pos="0.15 0 0" mass="0.5" diaginertia="0.02 0.02 0.02"/>
                <joint name="joint2" type="hinge" axis="0 1 0" range="-90 90" damping="0.1"/>
                <body name="link2" pos="0.3 0 0">
                    <geom name="link2_geom" type="capsule" fromto="0 0 0 0.3 0 0" size="0.04" rgba="0.2 0.8 0.2 1"/>
                    <inertial pos="0.15 0 0" mass="0.3" diaginertia="0.01 0.01 0.01"/>
                    <joint name="joint3" type="hinge" axis="0 1 0" range="-90 90" damping="0.05"/>
                    <body name="end_effector" pos="0.3 0 0">
                        <geom name="ee_geom" type="sphere" size="0.03" rgba="0.8 0.8 0.2 1"/>
                        <inertial pos="0 0 0" mass="{payload_mass:.9f}" diaginertia="{payload_inertia:.9f} {payload_inertia:.9f} {payload_inertia:.9f}"/>
                        <site name="ee_site" pos="0 0 0" size="0.01"/>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
    <actuator>
        <motor name="motor1" joint="joint1" gear="20"/>
        <motor name="motor2" joint="joint2" gear="20"/>
        <motor name="motor3" joint="joint3" gear="10"/>
    </actuator>
</mujoco>
"""
