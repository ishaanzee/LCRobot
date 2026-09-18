# Live Character Lamp

This repository implements a live character around the supplied five-DOF lamp
robot. The character uses the laptop camera, microphone, and speaker to detect
engagement, mirror arm motion, remember visible objects, respond to short
spoken requests, and execute simple scene-grounded lamp actions.

## Ubuntu 24.04 setup

```bash
sudo apt update
sudo apt install -y python3-venv portaudio19-dev libgl1 libglib2.0-0
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

The first run may take longer while model weights initialize. Set
`GROQ_API_KEY` in the environment or in a project-root `.env` file before using
speech features.

## Runtime controls

- `V`: record a three-second spoken request
- `A`: switch between right- and left-arm imitation
- `Q`: quit
- open palm: turn the lamp light on
- closed fist: turn the lamp light off

## System behavior

The camera pipeline runs engagement detection, pose tracking, hand gesture
recognition, and object detection. Pose landmarks drive the simulated lamp
joints during normal interaction. Open-palm and closed-fist gestures control the
light. Detected objects are stored in short-term scene memory so the character
can answer simple recall questions and ground spoken goals such as finding a
visible object.

Camera perception, pose tracking, simulation, sound effects, and generated
music run locally. A fixed three-second microphone recording is sent to Groq for
transcription, and the response text is sent to Groq for speech generation.
Camera frames are not sent to Groq. The planner uses a small deterministic
action vocabulary for lamp direction and light state.

## Verification

```bash
python -m unittest discover -s tests -v
```

The automated suite covers arm geometry, robot joint limits, gesture handling,
speech configuration, scene memory, language planning, and action execution.
Run a live check before submission as well, since camera placement, room
lighting, microphone input, speaker output, network latency, and actual CPU/RAM
usage depend on the deployment machine.

Architecture details, measurements, tradeoffs, and known limitations are in
[`TECHNICAL_NOTE.md`](TECHNICAL_NOTE.md).
