"""Dimensionally explicit flow and gradient diagnostics; no assumed Van Deemter optimum."""
import math


def linear_velocity_mm_s(flow_ml_min, diameter_mm, porosity=None):
    if not math.isfinite(flow_ml_min) or not math.isfinite(diameter_mm) or flow_ml_min <= 0 or diameter_mm <= 0:
        raise ValueError("Flow and diameter must be positive and finite")
    if porosity is not None and (not math.isfinite(porosity) or not 0 < porosity <= 1):
        raise ValueError("Porosity must be in (0,1]")
    superficial = flow_ml_min*1000/60 / (math.pi*diameter_mm**2/4)
    return superficial if porosity is None else superficial/porosity


def bed_volume_ml(length_mm, diameter_mm):
    if length_mm <= 0 or diameter_mm <= 0:
        raise ValueError("Invalid column geometry")
    return math.pi*diameter_mm**2/4*length_mm/1000


def gradient_segments(program):
    segments = []
    start_b = float(program[0]["B [%]"])
    maximum = max(float(p["B [%]"]) for p in program)
    returned = False
    for a,b in zip(program, program[1:]):
        t0,t1 = float(a["t [min]"]),float(b["t [min]"])
        duration = t1-t0
        if duration <= 0:
            raise ValueError("Gradient times must increase")
        b0,b1 = float(a["B [%]"]),float(b["B [%]"])
        slope = (b1-b0)/duration
        if slope < 0:
            role = "return_to_initial"
            returned = True
        elif b0 == b1 == start_b and returned:
            role = "re_equilibration"
        elif b0 == b1 == maximum and maximum > start_b:
            role = "high_organic_wash"
        else:
            role = "analytical_ramp_or_initial_hold"
        volume = duration*(float(a["flow rate [ml/min]"])+float(b["flow rate [ml/min]"]))/2
        segments.append({"start_min": t0,"end_min": t1,"duration_min": duration,
                         "b_slope_percentage_points_min": slope,"volume_ml": volume,"role": role})
    return segments


def hydrodynamic_features(row):
    """No pore velocity unless porosity is explicitly provided in metadata."""
    meta = row["metadata"]
    nan = float("nan")
    try:
        diameter = float(meta["column.id"])
        flow = float(meta["column.flowrate"])
        u = linear_velocity_mm_s(flow, diameter)
    except (ValueError, TypeError, KeyError):
        return [nan]*7
    porosity = meta.get("column.total_porosity")
    try:
        pore_u = linear_velocity_mm_s(flow,diameter,float(porosity)) if porosity not in (None,"") else nan
    except (ValueError, TypeError):
        pore_u = nan
    try:
        segments = gradient_segments(row["gradient_program"])
        total = sum(s["duration_min"] for s in segments)
        mean_u = linear_velocity_mm_s(sum(s["volume_ml"] for s in segments)/total,diameter)
        positive_slope = max(0,max(s["b_slope_percentage_points_min"] for s in segments))
        durations = [sum(s["duration_min"] for s in segments if s["role"] == role) for role in
                     ("analytical_ramp_or_initial_hold","high_organic_wash","re_equilibration")]
        return [u,pore_u,mean_u,positive_slope,*durations]
    except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError):
        return [u,pore_u,nan,nan,nan,nan,nan]
