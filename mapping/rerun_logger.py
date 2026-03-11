"""
Mapping Rerun Logger. Sets up the experiment blueprint and logs the map and robot position.
"""
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from mapping import OneMap
from onemap_utils import log_map_rerun


def log_pos(x, y, agent):
    agents_colors = [[255,0,0],[0,255,0],[0,0,255]]
    rr.log(f"map/agent_{agent}/position", rr.Points2D(rotate_frame([[x, y]]), colors=[agents_colors[agent]], radii=[2]))

def rotate_frame(points):
    return [[y, x] for (x, y) in points]

def setup_blueprint_debug(n_agents):
    my_blueprint = rrb.Blueprint(
        rrb.Horizontal(
            *[rrb.Vertical(
                rrb.Spatial2DView(
                    origin=f"agent_{i}/camera",
                    name=f"agent_{i}_rgb",
                    contents=["$origin/rgb", "$origin/detection", "$origin/target"], 
                ),
                rrb.Spatial2DView(origin=f"agent_{i}/camera/depth", name=f"agent_{i}_depth")
            ) for i in range(n_agents)],
            rrb.Vertical(
                rrb.Vertical(
                    rrb.TextLogView(origin="object_detections"),
                    rrb.TextLogView(origin="path_updates"),
                ),
            ),
            rrb.Vertical(
                rrb.Tabs(
                    *[rrb.Spatial2DView(origin="map",
                                        name="Similarity",
                                        contents=
                                        ["$origin/similarity/",
                                         "$origin/frontiers",
                                        *[f"$origin/agent_{i}/proj_detect" for i in range(n_agents)],
                                        *[f"$origin/agent_{i}/position" for i in range(n_agents)]]),
                      rrb.Spatial2DView(origin="map",
                                        name="SimilarityTresholded",
                                        contents=
                                        ["$origin/similarity_th/",
                                         *[f"$origin/agent_{i}/proj_detect" for i in range(n_agents)],
                                         *[f"$origin/agent_{i}/position" for i in range(n_agents)]]),
                      rrb.Spatial2DView(origin="map",
                                        name="SimilarityTresholdedCl",
                                        contents=
                                        ["$origin/similarity_th2/",
                                         *[f"$origin/agent_{i}/proj_detect" for i in range(n_agents)],
                                         *[f"$origin/agent_{i}/position" for i in range(n_agents)]]),
                      ],
                ),
                rrb.Tabs(
                    rrb.Spatial2DView(origin="map",
                                      name="Explored",
                                      contents=["$origin/explored",
                                                "$origin/largest_contour",
                                                "$origin/ground_truth",
                                                "$origin/frontiers",
                                                *[f"$origin/agent_{i}/position" for i in range(n_agents)],
                                                *[f"$origin/agent_{i}/goal_pos" for i in range(n_agents)],
                                                *[f"$origin/agent_{i}/path" for i in range(n_agents)],
                                                *[f"$origin/agent_{i}/path_simplified" for i in range(n_agents)],
                                                *[f"$origin/agent_{i}/proj_detect" for i in range(n_agents)],
                                            ]),
                    rrb.Spatial2DView(origin="map",
                                      name="Unexplored",
                                      contents=["$origin/frontiers",
                                                "$origin/largest_contour",
                                                "$origin/unexplored",
                                                *[f"$origin/agent_{i}/position" for i in range(n_agents)]]),
                ),
            ),
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(my_blueprint)

def setup_blueprint(n_agents):
    my_blueprint = rrb.Blueprint(
        rrb.Horizontal(
            *[rrb.Vertical(
                rrb.Spatial2DView(
                    origin=f"agent_{i}/camera",
                    name=f"agent_{i}_rgb",
                    contents=["$origin/rgb", "$origin/detection", "$origin/target"], 
                ),
                rrb.Spatial2DView(origin=f"agent_{i}/camera/depth", name=f"agent_{i}_depth")
            ) for i in range(n_agents)],
            rrb.Vertical(
                rrb.Tabs(
                    rrb.Spatial2DView(
                        origin="map",
                        name="Similarity",
                        contents=["$origin/similarity/", *[f"$origin/agent_{i}/position" for i in range(n_agents)]]
                    ),
                ),
                rrb.Tabs(
                    rrb.Spatial2DView(
                        origin="map",
                        name="Explored",
                        contents=[
                            "$origin/explored",
                            *[f"$origin/agent_{i}/position" for i in range(n_agents)],
                            *[f"$origin/agent_{i}/goal_pos" for i in range(n_agents)],
                            *[f"$origin/agent_{i}/path" for i in range(n_agents)],
                            *[f"$origin/agent_{i}/path_simplified" for i in range(n_agents)],
                        ]
                    ),
                ),
            ),
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(my_blueprint)

class RerunLogger:
    def __init__(self, one_map: OneMap, to_file: bool, save_path: str, n_agents: int, debug: bool = True):
        self.debug_log = debug
        self.to_file = to_file
        self.one_map = one_map
        self.n_agents = n_agents

        rr.init("MON", spawn=False)
        if self.to_file:
            rr.save(save_path)
        else:
            rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")

        if self.debug_log:
            setup_blueprint_debug(self.n_agents)
        else:
            setup_blueprint(self.n_agents)

        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # Set an up-axis
        rr.log(
            "world/xyz",
            rr.Arrows3D(
                vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )

    def log_map(self):
        confidences = self.one_map.confidence_map.cpu().numpy()
        similarities = (self.one_map.get_similarity_map() + 1.0) / 2.0

        explored = (self.one_map.navigable_map == 1).astype(np.float32) * 0.1
        explored[confidences > 0] = 0.5
        explored[self.one_map.fully_explored_map] = 1.0
        explored[self.one_map.navigable_map == 0] = 0

        log_map_rerun(explored, path="map/explored")
        log_map_rerun(similarities[0], path="map/similarity")

    def log_pos(self, x, y, agent):
        px, py = self.one_map.metric_to_px(x, y)
        log_pos(px, py, agent)
