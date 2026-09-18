import re


def plan_request(text, visible_objects, remembered_objects):
    """Turn a short spoken request into a reply and simple lamp actions."""
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    singular_words = {word[:-1] if word.endswith("s") else word for word in words}
    known_labels = set(visible_objects) | set(remembered_objects)
    target = next(
        (label for label in sorted(known_labels, key=len, reverse=True)
         if set(label.lower().split()) <= singular_words),
        None,
    )

    asking_about_scene = bool(words & {"what", "remember", "remembered", "saw", "see"})
    requesting_action = bool(words & {"find", "look", "point", "turn"})

    if asking_about_scene and not requesting_action:
        if target and target in remembered_objects:
            position = remembered_objects[target]["position"]
            return [], f"I remember the {target} on the {position}.", None
        if remembered_objects:
            recent_objects = sorted(
                remembered_objects.items(),
                key=lambda item: item[1]["last_seen"],
                reverse=True,
            )[:4]
            summary = ", ".join(
                f"a {label} on the {item['position']}"
                for label, item in recent_objects
            )
            return [], f"I remember {summary}.", None
        return [], "I have not remembered any objects yet.", None

    if requesting_action and target:
        if target not in visible_objects:
            position = remembered_objects[target]["position"]
            return [], (
                f"I remember the {target} on the {position}, "
                "but I cannot see it now."
            ), None
        position = visible_objects[target]["position"]
        actions = [("look", position)]
        if "light" in words or "lamp" in words:
            actions.append(("light", "off" if "off" in words else "on"))
        return actions, "", target

    if "light" in words or "lamp" in words:
        if "off" in words:
            return [("light", "off")], "Turning the light off.", None
        if "on" in words:
            return [("light", "on")], "Turning the light on.", None

    if target:
        observation = remembered_objects.get(target, visible_objects[target])
        position = observation["position"]
        return [], f"The {target} is on the {position}.", None
    return [], (
        "Try asking what I remember, or tell me to find a visible object "
        "and turn on my light."
    ), None


def perform_actions(lamp, actions):
    """Execute the small action vocabulary produced by plan_request."""
    for action, value in actions:
        if action == "look":
            if value == "left":
                lamp.look_left()
            elif value == "right":
                lamp.look_right()
            else:
                lamp.look_center()
        elif action == "light":
            lamp.turn_on() if value == "on" else lamp.turn_off()
