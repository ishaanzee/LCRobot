# Arm following

Run `python main.py` in the environment with the project dependencies installed.
The tracker starts with your **right arm**. Press **A** to switch arms and **Q**
to quit. Keep the selected shoulder, elbow, and wrist in view. A green dot marks
its shoulder, a blue dot marks its elbow, and a yellow dot marks its wrist.
The camera image is not mirrored.

## Joint mapping

| Your movement | Lamp joint |
| --- | --- |
| Sweep the upper arm around a vertical axis | Base rotation |
| Raise or lower the upper arm | Shoulder tilt above the base |
| Bend or straighten the elbow | Middle pivot |

The base and shoulder joint together approximate two rotations of your shoulder.
The wrist only contributes to the elbow angle, so moving your forearm while
holding your upper arm still leaves the lamp's bottom two joints alone. The
neck and head stay neutral relative to the upper lamp segment during imitation.

## How to explain it in an interview

1. MediaPipe gives us both image landmarks and estimated 3D world landmarks.
   Image landmarks check visibility and draw the arm overlay. World landmarks
   provide positions in meters, so the angle calculations use the same scale
   along all three axes. See the [MediaPipe pose documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker).
2. Subtract shoulder position from elbow position to get the upper-arm vector.
   Subtract elbow position from wrist position to get the forearm vector.
   Translation cancels out: moving your entire body does not change the angles.
3. In camera coordinates, x points image-right, y down, and z away from the
   camera. `atan2(-upper_x, -upper_z)` measures horizontal direction. Reaching
   toward the camera means lamp-forward; reaching toward image-left means
   lamp-left. This uses a fixed camera frame, so keep the camera upright.
4. `atan2(-upper_y, hypot(upper_x, upper_z))` measures upper-arm elevation:
   down is -90 degrees, horizontal is 0, and overhead is +90 degrees.
5. The dot product of the upper-arm and forearm vectors, divided by their
   lengths, gives the cosine of elbow bend. `acos` converts it to radians:
   straight is 0, a right-angle elbow is about 1.57.
6. One exponential filter smooths the three angles. Its weight is
   `1 - exp(-elapsed_time / 0.08)`, giving the same response at different frame
   rates. It covers about 92% of a steady change in 0.2 seconds. This describes
   the filter alone, not measured camera-to-display latency.
7. `LampController.follow_arm` copies yaw to the base and directly matches the
   reachable upper-arm elevation. Because the lamp segment is vertical when its
   shoulder joint is zero, `shoulder_pitch = pi/2 - arm_elevation`. It also sets
   `elbow_pitch = -elbow_bend`. MuJoCo uses forward kinematics to draw the pose.

## Range and stability choices

A human shoulder can move farther than this lamp. Within the lamp's reachable
range, its bottom segment makes the same angle with the ground as your upper
arm. The conversion is:

`shoulder_pitch = clamp(pi/2 - arm_elevation, 0, 0.95)`

| Your upper arm | Bottom lamp segment |
| --- | --- |
| 90 degrees above ground | 90 degrees above ground |
| 45 degrees above ground | 45 degrees above ground |
| Below about 36 degrees | Stops at about 36 degrees |

The final case respects the URDF shoulder soft limit of 0.95 radians. The lamp
cannot become horizontal or point through the table, so lower human-arm angles
saturate instead of being rescaled or extrapolated.
Base rotation stops at +/-2.45 radians and elbow bend at 1.7 radians, within
the URDF soft limits. These range choices are visible directly in the code.

A nearly vertical upper arm has no clear horizontal direction. When its
horizontal component is less than 15% of its length, hold the last base yaw
(or face forward if there is no recent pose). When the measured yaw crosses
+/-180 degrees behind you, choose the nearest equivalent angle before limiting
it. This prevents jitter from sending the base spinning through the front.

If the selected shoulder, elbow, or wrist is hidden/offscreen, or the 3D
coordinates are invalid or have segments shorter than 5 cm, hold the last pose.
After tracking has been lost for half a second, start from the new angles rather
than blending with stale ones. Reacquisition can therefore produce a jump.

Object recognition continues in one background worker, with at most one job
outstanding and new submissions no more often than every 0.5 seconds. The main
loop can keep tracking while the object model runs, although they still share
CPU resources. Reused object boxes can briefly trail moving objects.

## Checks and limitations

Run `python -m unittest discover -s tests -v`. Tests cover known 3D arm directions,
independent shoulder/elbow motion, both arms, translation/scale independence,
invalid landmarks, smoothing, reacquisition, vertical-arm stability, rear angle
wrapping, and movement direction and joint limits in the actual MuJoCo model.

For a live check:

1. Keep the elbow bent and sweep your upper arm from sideways toward the camera.
   The lamp base should rotate.
2. Raise and lower the upper arm without changing elbow bend. The bottom lamp
   segment should tilt, while the middle joint keeps its bend angle.
3. Hold the upper arm still and bend/straighten the elbow. Only the middle joint
   should change.
4. Raise the arm nearly vertical; the base should stop chasing tiny sideways
   movements. Hide the elbow briefly, recover, and try the other arm with A.

Single-camera depth is an estimate, so occlusion, turning away, and pointing a
limb at the lens can still cause errors. There is no shoulder twist/roll mapping:
the lamp has only base yaw and shoulder pitch for that part of the motion. Elbow
bend is copied in magnitude into the lamp's fixed bending plane, so forearm
rotation is not reproduced. Ground elevation assumes an upright camera rather
than estimating a floor plane. The simulation sets joint positions directly;
it does not enforce motor speed, acceleration, or physical actuator dynamics.
Live webcam validation is still needed to judge responsiveness on your laptop.
