"""Write explicit, reproducible hypotheses, keeping observations separate."""
import argparse
import copy
import json
from pathlib import Path


SOURCES = {
    "zhou2025": {
        "doi": "10.3847/1538-4357/adcf18",
        "url": "https://iopscience.iop.org/article/10.3847/1538-4357/adcf18",
        "locator": "Sections 4.1-4.4; Figure 4(d-f); Appendix Figure A6",
        "evidence_type": "radar_interpretation_and_forward_model",
        "reported_estimates": {"maximum_width_m_approx": 26, "maximum_height_m_approx": 16,
                               "burial_depth_m_approx": 130},
        "limits": ["Void interpretation is non-unique.",
                   "Width may be overestimated by an oblique traverse.",
                   "No measured axis, longitudinal extent, branches or entrance.",
                   "Figure A6 is a forward model, not a surveyed mesh."]},
    "tomasi2022": {
        "doi": "10.1029/2022JB024056",
        "url": "https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022JB024056",
        "evidence_type": "terrestrial_laser_survey_and_discussed_formation_analogue",
        "locator": "Sections 2.1.3, 3.2 and 4.1; Figures 4-5",
        "use": "Qualitative side branches, rejoining passages and superposed levels.",
        "evidence_distinction": "Inter-level windows and isolated upper conduits are identified in 3D; side-tube rejoining is also discussed as a formation interpretation, not a mapped lunar connection.",
        "limits": ["Earth geometry is not a lunar survey.",
                   "No calibrated probability of lunar branching or loop counts.",
                   "Absolute dimensions and positions are not transferred."]},
}


def edge(eid, start, end, points, left=10., right=16., height=16., role="main"):
    return {"id": eid, "from": start, "to": end,
            "floor_centerline_xyz_m": points,
            "cross_section": {"left_width_m": left, "right_width_m": right,
                              "height_m": height, "roof_offset_m": -1.5,
                              "roof_sag_m": 2.4 if role == "main" else (1.6 if role == "rejoining_branch" else 1.2)},
            "role": role,
            "provenance": {
                "topology": "generation_assumption" if role == "main" else "earth_analogue_inspired_hypothesis",
                "coordinates_and_length": "generation_assumption",
                "source_ids": ["zhou2025"] if role == "main" else ["zhou2025", "tomasi2022"],
                "cross_section": "parameterized_hypothesis_inspired_by_zhou2025",
                "note": "No edge is an observed lunar passage. Widths are maximum lateral radii, not floor clearances."}}


def candidates():
    pos = {"A": [0, 0, -146], "B": [100, 5, -146],
           "C": [220, 0, -148], "D": [350, 20, -151],
           "E": [155, 85, -154], "F": [290, -65, -188]}
    main = [edge("main_ab", "A", "B", [pos["A"], [35,-4,-147], [65,3,-146], pos["B"]]),
            edge("main_bc", "B", "C", [pos["B"], [140,18,-145.5], [180,15,-146.5], pos["C"]]),
            edge("main_cd", "C", "D", [pos["C"], [260,2,-149], [305,8,-150], pos["D"]])]
    loop = [edge("loop_be", "B", "E", [pos["B"], [110,45,-150], [125,72,-153], pos["E"]], 7,11,14,"rejoining_branch"),
            edge("loop_ec", "E", "C", [pos["E"], [195,80,-155], [225,50,-153], pos["C"]], 7,11,14,"rejoining_branch")]
    lower = edge("lower_ef", "E", "F", [pos["E"], [220,115,-161], [295,90,-173],
                  [300,45,-181], [260,2,-187], [270,-40,-188], pos["F"]], 5.5,8.5,12,"lower_level_branch")
    common = {
        "schema_version": "lunar_candidate_network/1.0", "units": "m", "up_axis": "Z",
        "coordinate_system": "local arbitrary horizontal axes; z=0 is an assumed reference lunar surface",
        "random_seed": 20261110, "voxel_spacing_m": 1.0,
        "scientific_status": "synthetic_candidate_not_a_reconstruction",
        "site_identity": "Zhou-scale hypothetical underground network; not MTP or Marius Hills",
        "sources": SOURCES,
        "parameter_interpretation": {
            "floor_centerline_xyz_m": "Local support/floor reference; each branch retains its own z.",
            "cross_section": "Asymmetric rounded void with offset arch and roof indentation; procedural, not digitized from Figure A6.",
            "width_m": "Nominal main-section maximum 26 m is inspired by the reported estimate, not a lower clearance bound. Junction unions can be wider.",
            "height_m": "Nominal main-section floor-to-roof upper envelope 16 m; roof indentation reduces it locally.",
            "depth": "Main roof placed roughly near -130 m as a visualization assumption; the paper does not establish this exact floor or roof datum.",
            "roughness": "No fabricated sub-metre measured lunar surface detail.",
            "roof_sag_m": "Authored visual-shape hypothesis: main 2.4 m, rejoining branch 1.6 m, lower branch 1.2 m; not measured by the cited radar study.",
            "topology": "All longitudinal coordinates, lengths, turns, branch locations and level spacings are assumptions.",
            "end_caps": "Artificial finite-domain closures; do not imply observed geological dead ends.",
            "entry": "No observed skylight is assigned; the underground model has no surface entry.",
            "stability": "No geomechanical solution or structural safety certification.",
            "probability": "Candidates are unweighted hypotheses; no inferred posterior probabilities."},
        "robot_envelope": {"radius_m": .4, "height_m": 1., "max_slope_deg": 12.,
                           "classification": "engineering_assumption_not_lunar_observation"},
        "runtime_scope": "geometry_only; old XY navigation and floor_height are not connected",
        "nonconnecting_crossings": [],
    }
    specs = [("01_trunk", "Single sinuous trunk", main, ["A","B","C","D"]),
             ("02_rejoining", "Trunk with one rejoining side passage", main+loop, ["A","B","C","D","E"]),
             ("03_multilevel", "Rejoining network with a separated lower passage", main+loop+[lower], list(pos))]
    result = []
    for cid, title, edges, node_ids in specs:
        network = copy.deepcopy(common)
        network.update(candidate_id=cid, title=title, edges=copy.deepcopy(edges),
                       nodes=[{"id": n, "xyz_m": pos[n],
                               "kind": "synthetic_domain_end" if n in ("A","D","F") else "synthetic_junction",
                               "provenance": "generation_assumption"} for n in node_ids])
        if cid == "03_multilevel":
            network["nonconnecting_crossings"] = [{"edge_a": "main_cd", "edge_b": "lower_ef",
                  "xy_m": [260,2], "min_rock_thickness_m": 12.,
                  "provenance": "generation_assumption; projected crossing deliberately remains separate"}]
        result.append(network)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent/"configs/paper_network_v3")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for network in candidates():
        path = args.output / (network["candidate_id"] + ".json")
        path.write_text(json.dumps(network, ensure_ascii=False, indent=2)+"\n")
        print(path)


if __name__ == "__main__":
    main()
