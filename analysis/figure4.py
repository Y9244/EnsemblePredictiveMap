import matplotlib.pyplot as plt

from config.default_config import (
    SquareArenaConfig, RLConfig, GridCellConfig, BoundaryCellConfig,
    SparseCodingConfig, PmapConfig)
from envs.square_arena import SquareArenaEnv
from models.grid_cells import GridCell
from models.boundary_cell import BoundaryCell
from models.sparse_coding import SparseCoding
from models.predictive_map import Pmap
from models.RL_module import RL
from visualizer.visualize_map import VisualizeMap
from models.wrapper import one_step_grid
from utils.logger import Logger
import pickle
import numpy as np
from visualizer.metrics import show_predictive_accuracy

if __name__ == "__main__":
    with open('../data/Frontiers/figure4.pkl', 'rb') as f:
        figure4 = pickle.load(f)

    sparse_coding = figure4['sparse_coding']
    pmap = figure4['pmap']
    agent = figure4['agent']
    visualize_map = figure4['visualize_map']
    logger = figure4['logger']


    sc_place_fields = visualize_map.compute_sc_place_fields(sparse_coding)
    pmap_place_fields = visualize_map.compute_pmap_place_fields(pmap, sc_place_fields)
    place_centers, place_center_index = visualize_map.compute_place_centers_in_square_arena(sc_place_fields,
                                                                                            return_index=True)
    Vmap = visualize_map.compute_Vmap(pmap_place_fields, agent.R, agent.reward_weights)

    print("Figure4A")
    visualize_map.show_figure4ac(sc_place_fields, base_dir='Frontiers', name='Figure4A')
    print("Figure4B")
    visualize_map.show_place_density_in_square_arena(place_centers, place_center_index,
                                                     reward_positions=False, reward_or_object='reward', base_dir='Frontiers', name='Figure4B')
    print("Figure4C")
    visualize_map.show_figure4ac(np.where(pmap_place_fields > 0, pmap_place_fields, 0), base_dir='Frontiers', name='Figure4C')

    print("Figure4D")
    visualize_map.show_peak_mass_vector(sc_place_fields, pmap_place_fields, place_center_index, base_dir="Frontiers", name="Figure4D")

    print("Figure4E")
    visualize_map.show_peak_mass_dist(sc_place_fields, pmap_place_fields, place_center_index,
                                      display='bar', significance=True, base_dir="Frontiers", name='Figure4E')

    print("Figure4F")
    visualize_map.show_predictive_accuracy(logger.sc_recordings, logger.pm_recordings, display='bar', name='Figure4F')

    print("Figure4GH")
    visualize_map.show_figure5ab(place_centers, place_center_index, agent.R, Vmap, name='Figure4GH')


