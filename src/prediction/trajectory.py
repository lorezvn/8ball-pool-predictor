"""Geometric shot prediction in the rectified top-down view.

Everything here is pure geometry on the canonical CANONICAL_W x CANONICAL_H
rectangle, where cushions are axis-aligned walls and a bounce is a plain
reflection. No OpenCV, no YOLO: the caller projects the detections into this
space and projects the resulting paths back.

The model is the standard "ghost ball" approximation:
  - the cue ball travels along the stick direction until something happens;
  - two balls of radius R touch when their centers are 2R apart;
  - the struck ball leaves along the line of centers;
  - the cue ball leaves along the tangent, perpendicular to it (90 degrees rule).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from detection.detect_table import CANONICAL_W, CANONICAL_H

# Cushion bounces allowed inside a single link. 0 means a clean straight line,
# which is the only regime that is actually meaningful today: without a friction
# model a ball never loses speed, so with bounces enabled the IN/OUT verdict is
# decided by how many bounces you allow rather than by the shot itself.
MAX_BOUNCES = 0
MAX_DEPTH = 3           # how many balls deep the collision chain may go
# A pocket captures a ball whose centre arrives within CAPTURE_FACTOR * radius.
# Calibrated by sweeping this value against the real outcome of the sample clips
# (which ball actually vanished from the table): 2.2 scored 4/9, 3.2 scores 6/9
# and nothing above it improves. A real pocket mouth is wider than a ball, so a
# ball passing ~30 px from the centre does drop; the inherited 2.2 was too strict.
# The sample is small -- 9 clips, of which two pairs are the same shot -- so treat
# this as a reasonable setting, not a solid calibration.
CAPTURE_FACTOR = 3.2

# In the canonical view the pockets are exactly the 4 corners plus the midpoints
# of the two long edges, so they need no detection.
POCKETS_TOP = [
    (0.0, 0.0),
    (CANONICAL_W, 0.0),
    (CANONICAL_W, CANONICAL_H),
    (0.0, CANONICAL_H),
    (CANONICAL_W / 2, 0.0),
    (CANONICAL_W / 2, CANONICAL_H),
]


def nearest_wall(point: np.ndarray, direction: np.ndarray, radius: float) -> tuple:
    """Distance to the first cushion hit, and which axis it reflects.

    The playable area is inset by `radius`, because it is the ball centre that
    travels and the ball touches the cushion one radius early.

    Returns:
        (t, axis) with axis in {"x", "y"}, or (None, None) if the ray escapes.
    """
    x_min, x_max = radius, CANONICAL_W - radius
    y_min, y_max = radius, CANONICAL_H - radius

    hits = []
    if direction[0] > 1e-9:
        hits.append(((x_max - point[0]) / direction[0], "x"))
    if direction[0] < -1e-9:
        hits.append(((x_min - point[0]) / direction[0], "x"))
    if direction[1] > 1e-9:
        hits.append(((y_max - point[1]) / direction[1], "y"))
    if direction[1] < -1e-9:
        hits.append(((y_min - point[1]) / direction[1], "y"))

    hits = [(t, axis) for t, axis in hits if t > 1e-6]
    return min(hits) if hits else (None, None)


def first_ball_hit(
    point: np.ndarray,
    direction: np.ndarray,
    radius: float,
    balls: list,
    skip: set,
    t_max: float,
) -> tuple:
    """First ball met along point + t * direction, within t_max.

    Contact happens when the centres are 2*radius apart, which is a ray/circle
    intersection against a circle of radius 2R centred on the target ball.

    Returns:
        (t, index) of the closest hit, or (None, None).
    """
    best_t, best_i = None, None

    for i, centre in enumerate(balls):
        if i in skip:
            continue

        offset = point - centre
        b = float(offset @ direction)
        c = float(offset @ offset) - (2 * radius) ** 2
        discriminant = b * b - c
        if discriminant < 0:
            continue

        t = -b - np.sqrt(discriminant)   # first root: the approaching contact
        if 1e-6 < t < t_max and (best_t is None or t < best_t):
            best_t, best_i = t, i

    return best_t, best_i


def propagate(
    point: np.ndarray,
    direction: np.ndarray,
    radius: float,
    balls: list,
    skip: set,
    capture: float,
    max_bounces: int = 0,
) -> tuple:
    """Advance one ball until the first event, bouncing off cushions on the way.

    Returns:
        (path, event) where path is the list of points travelled and event is
        ("ball", index, contact, struck_direction) | ("pocket",) | ("none",).
    """
    pockets = [np.array(p, dtype=np.float64) for p in POCKETS_TOP]
    point = point.astype(np.float64)
    direction = direction / (np.linalg.norm(direction) + 1e-9)
    path = [point.copy()]

    for _ in range(max_bounces + 1):
        t_wall, axis = nearest_wall(point, direction, radius)
        if t_wall is None:
            break

        t_ball, j = first_ball_hit(point, direction, radius, balls, skip, t_wall)
        if t_ball is not None:
            contact = point + direction * t_ball
            path.append(contact)
            struck = balls[j] - contact                     # line of centres
            return path, ("ball", j, contact, struck / (np.linalg.norm(struck) + 1e-9))

        hit = point + direction * t_wall
        path.append(hit)
        if min(np.linalg.norm(hit - p) for p in pockets) <= capture:
            return path, ("pocket",)

        # reflect off the cushion and carry on
        direction = np.array([-direction[0], direction[1]]) if axis == "x" \
            else np.array([direction[0], -direction[1]])
        point = hit

    return path, ("none",)


def simulate_chain(
    start: np.ndarray,
    direction: np.ndarray,
    radius: float,
    balls: list,
    capture: float,
    max_depth: int = MAX_DEPTH,
    max_bounces: int = 0,
) -> tuple:
    """Follow the chain cue ball -> ball A -> ball B -> ... up to max_depth.

    Returns:
        (segments, potted_index) where segments is a list of (path, ball_index)
        with ball_index None for the cue ball's own leg, and potted_index is the
        index of the ball that drops (None if none, -1 if the cue ball scratches).
    """
    segments = []
    skip: set = set()
    point, current = start, None            # current None means "the cue ball"
    potted_index = None

    for _ in range(max_depth + 1):
        path, event = propagate(point, direction, radius, balls, skip, capture, max_bounces)
        segments.append((path, current))

        if event[0] == "pocket":
            potted_index = current if current is not None else -1
            break

        if event[0] == "ball":
            j = event[1]
            skip.add(j)
            point, direction, current = balls[j], event[3], j
            continue

        break                                # the ball rolls to a stop

    return segments, potted_index


def cue_deflection(
    segments: list,
    balls: list,
    radius: float,
    capture: float,
    max_bounces: int = 0,
) -> tuple:
    """Where the cue ball goes after its first contact (the 90 degrees rule).

    It leaves along the tangent: the component of its incoming direction that is
    perpendicular to the line of centres.

    Returns:
        (path, scratches) or (None, False) if there was no ball contact.
    """
    if len(segments) < 2 or segments[0][1] is not None or segments[1][1] is None:
        return None, False

    cue_path = segments[0][0]
    contact = cue_path[-1]

    incoming = contact - cue_path[-2]
    incoming = incoming / (np.linalg.norm(incoming) + 1e-9)

    normal = balls[segments[1][1]] - contact          # line of centres
    normal = normal / (np.linalg.norm(normal) + 1e-9)

    tangent = incoming - float(incoming @ normal) * normal
    if np.linalg.norm(tangent) <= 1e-6:               # dead straight hit: no deflection
        return None, False

    tangent = tangent / np.linalg.norm(tangent)
    path, event = propagate(
        contact, tangent, radius, balls, {segments[1][1]}, capture, max_bounces
    )
    return path, event[0] == "pocket"
