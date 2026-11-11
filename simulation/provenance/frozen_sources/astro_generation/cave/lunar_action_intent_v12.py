"""Coarse language actions for actual executed rover controls."""


def action_intent(v, omega, stop, motion_controller='segment'):
    if stop:
        return 'stop'
    turning_threshold = 1e-8
    if motion_controller.startswith('smooth') and v > 0.02:
        # Normal steering on a curved forward path is still forward motion.
        turning_threshold = 0.12
    if omega > turning_threshold:
        return 'turn_left'
    if omega < -turning_threshold:
        return 'turn_right'
    if v > 1e-8:
        return 'move_forward'
    return 'hold'
