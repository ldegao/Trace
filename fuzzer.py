#!/usr/bin/env python3
"""
MIT License

Copyright (c) 2024 [??????????]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import cProfile
import logging
import os
import pdb
import re
import sys
import time
import random
import argparse
import copyreg
from collections import OrderedDict

import gpt
import json
# from collections import deque
import concurrent.futures
import math
from types import SimpleNamespace
from typing import List
from subprocess import Popen, PIPE

import docker
import numpy as np
import torch
from deap import base, tools, algorithms
import signal
import traceback
import networkx as nx
from shapely.geometry import LineString
import config
import constants as c
from npc import NPC
from scenario import Scenario
import states
import utils

config.set_carla_api_path()
try:
    import carla
except ModuleNotFoundError as e:
    print("[-] Carla module not found. Make sure you have built Carla.")
    proj_root = config.get_proj_root()
    print("    Try `cd {}/carla && make PythonAPI' if not.".format(proj_root))
    exit(-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
client, world, G, blueprint_library, town_map = None, None, None, None, None
# model = cluster.FeatureExtractor().to(device)
accumulated_trace_graphs = []
autoware_container = None
exec_state = states.ExecState()
Scenario_database = {}
bottleneck = False

with open("api.json") as f:
    api_config = json.load(f)
API_KEY = api_config.get("OPENAI_API_KEY")
PROXY = {
    "http": "http://127.0.0.1:8080",
    "https": "http://127.0.0.1:8080"
}

prompt_file = "prompt.txt"
with open(prompt_file, 'r') as f:
    prompt = f.read()


# monitor carla
# monitoring_thread = utils.monitor_docker_container('carlasim/carla:0.9.13')
# vehicle_bp_library = blueprint_library.filter("vehicle.*")
# vehicle_bp.set_attribute("color", "255,0,0")
# walker_bp = blueprint_library.find("walker.pedestrian.0001")  # 0001~0014
# walker_controller_bp = blueprint_library.find('controller.ai.walker')
# player_bp = blueprint_library.filter('nissan')[0]


def create_test_scenario(conf, seed_dict):
    return Scenario(conf, seed_dict)


def handler(signum, frame):
    raise Exception("HANG")


def ini_hyperparameters(conf, args):
    conf.cur_time = time.time()
    if args.determ_seed:
        conf.determ_seed = args.determ_seed
    else:
        conf.determ_seed = conf.cur_time
    random.seed(conf.determ_seed)
    print("[info] determ seed set to:", conf.determ_seed)
    conf.out_dir = args.out_dir
    try:
        os.mkdir(conf.out_dir)
    except Exception:
        estr = f"Output directory {conf.out_dir} already exists. Remove with " \
               "caution; it might contain data from previous runs."
        print(estr)
        sys.exit(-1)

    conf.seed_dir = args.seed_dir
    if not os.path.exists(conf.seed_dir):
        os.mkdir(conf.seed_dir)
    else:
        print(f"Using seed dir {conf.seed_dir}")
    conf.set_paths()

    # with open(conf.meta_file, "w") as f:
    #     f.write(" ".join(sys.argv) + "\n")
    #     f.write("start: " + str(int(conf.cur_time)) + "\n")

    try:
        os.mkdir(conf.queue_dir)
        os.mkdir(conf.error_dir)
        os.mkdir(conf.picture_dir)
        os.mkdir(conf.rosbag_dir)
        os.mkdir(conf.cam_dir)
        os.mkdir(conf.npc_dir)
        os.mkdir(conf.time_record_dir)
    except Exception as e:
        print(e)
        sys.exit(-1)
    if args.no_lane_check:
        conf.check_dict["lane"] = False
    conf.sim_host = args.sim_host
    conf.sim_port = args.sim_port
    conf.max_mutations = args.max_mutations
    conf.timeout = args.timeout
    conf.function = args.function

    if args.target.lower() == "behavior":
        conf.agent_type = c.BEHAVIOR
    elif args.target.lower() == "autoware":
        conf.agent_type = c.AUTOWARE
    else:
        print("[-] Unknown target: {}".format(args.target))
        sys.exit(-1)

    conf.town = args.town
    conf.num_mutation_car = args.num_mutation_car
    conf.density = float(args.density)
    conf.no_traffic_lights = args.no_traffic_lights
    conf.debug = args.debug


def mutate_weather(test_scenario):
    test_scenario.weather["cloud"] = random.randint(0, 100)
    test_scenario.weather["rain"] = random.randint(0, 100)
    test_scenario.weather["wind"] = random.randint(0, 100)
    test_scenario.weather["fog"] = random.randint(0, 100)
    test_scenario.weather["wetness"] = random.randint(0, 100)
    test_scenario.weather["angle"] = random.randint(0, 360)
    test_scenario.weather["altitude"] = random.randint(-90, 90)


def mutate_weather_fixed(test_scenario):
    test_scenario.weather["cloud"] = 0
    test_scenario.weather["rain"] = 0
    test_scenario.weather["wind"] = 0
    test_scenario.weather["fog"] = 0
    test_scenario.weather["wetness"] = 0
    test_scenario.weather["angle"] = 0
    test_scenario.weather["altitude"] = 60


def set_args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--debug", action="store_true", default=False)
    argument_parser.add_argument("-o", "--out-dir", default="./data/output", type=str,
                                 help="Directory to save fuzzing logs")
    argument_parser.add_argument("-m", "--max-mutations", default=5, type=int,
                                 help="Size of the mutated population per cycle")
    argument_parser.add_argument("-d", "--determ-seed", type=float,
                                 help="Set seed num for deterministic mutation (e.g., for replaying)")
    argument_parser.add_argument("-u", "--sim-host", default="localhost", type=str,
                                 help="Hostname of Carla simulation server")
    argument_parser.add_argument("-p", "--sim-port", default=2000, type=int,
                                 help="RPC port of Carla simulation server")
    argument_parser.add_argument("-s", "--seed-dir", default="./data/seed", type=str,
                                 help="Seed directory")
    argument_parser.add_argument("-t", "--target", default="behavior", type=str,
                                 help="Target autonomous driving system (behavior/Autoware)")
    argument_parser.add_argument("-f", "--function", default="general", type=str,
                                 choices=["general", "collision", "traction", "eval-os", "eval-us",
                                          "figure", "sens1", "sens2", "lat", "rear"],
                                 help="Functionality to test (general / collision / traction)")
    argument_parser.add_argument("-k", "--num_mutation_car", default=3, type=int,
                                 help="Number of max weight vehicles to mutation per cycle, default=1,negative means "
                                      "random")
    argument_parser.add_argument("--density", default=1, type=float,
                                 help="density of vehicles,1.0 means add 1 bg vehicle per 1 sec")
    argument_parser.add_argument("--town", default=3, type=int,
                                 help="Test on a specific town (e.g., '--town 3' forces Town03)")
    argument_parser.add_argument("--timeout", default="60", type=int,
                                 help="Seconds to timeout if vehicle is not moving")
    argument_parser.add_argument("--no-speed-check", action="store_true")
    argument_parser.add_argument("--no-lane-check", action="store_true")
    argument_parser.add_argument("--no-crash-check", action="store_true")
    argument_parser.add_argument("--no-stuck-check", action="store_true")
    argument_parser.add_argument("--no-red-check", action="store_true")
    argument_parser.add_argument("--no-other-check", action="store_true")
    argument_parser.add_argument("--no-traffic-lights", action="store_true")
    return argument_parser


def extract_answer1_overall_similarity(response_text):
    """
    Extracts answer1 description and Overall Similarity from raw response text using regular expressions.

    Parameters:
    - response_text: The raw text of the JSON response

    Returns:
    - answer1_description: Extracted description of answer1
    - overall_similarity: Extracted similarity score (integer) or None if not found
    """
    # Regular expression to extract answer1 description
    answer1_match = re.search(r'"answer1":\s*\{\s*"Description":\s*"([^"]+)', response_text)
    answer1_description = answer1_match.group(1) if answer1_match else "Description not available"

    # Regular expression to extract Overall Similarity
    similarity_match = re.search(r'"Overall Similarity":\s*"(\d+)', response_text)
    overall_similarity = int(similarity_match.group(1)) if similarity_match else None

    return answer1_description, overall_similarity


def evaluation(ind: Scenario):
    global autoware_container
    global Scenario_database
    min_dist = 99999
    nova = 0
    overall_similarity = 0
    g_name = f'Generation_{ind.generation_id:05}'
    s_name = f'Scenario_{ind.scenario_id:05}'
    # run test here
    mutate_weather_fixed(ind)
    signal.alarm(15 * 60)  # timeout after 15 min
    print("timeout after 15 min")
    try:
        # profiler = cProfile.Profile()
        # profiler.enable()  #
        ind.state.scenario_id = ind.scenario_id
        ind.state.generation_id = ind.generation_id
        ret = ind.run_test(exec_state)
        if ret == -1:
            print("[-] Fatal error occurred during test")
            exit(0)
        min_dist = ind.state.min_dist

        for i in range(1, len(ind.state.speed)):
            acc = abs(ind.state.speed[i] - ind.state.speed[i - 1])
            nova += acc
        nova = nova / len(ind.state.speed)
        # gpt
        scenario_description = str(
            gpt.get_frame_data(f"./data/output/time_record/gid:{ind.generation_id}_sid:{ind.scenario_id}.json",
                               ind.state.min_dist_frame)).replace("\n", "").replace(' ', '')
        question = prompt + "\n scenario snapshot:\n" + str(
            scenario_description + "\n___\n Scenario-dict\n" + str(Scenario_database))
        response = gpt.call_gpt(question, model_version="gpt-4-turbo", max_tokens=1500)
        print("Response:", response)
        response_json = gpt.extract_json(response)
        if response_json is not None:
            Scenario_database = gpt.add_answer1_to_database(response_json, Scenario_database, 30)
            overall_similarity = int(gpt.get_overall_similarity(response_json))
            answer3_vehicle_info = gpt.get_answer3_vehicle_info(response_json)
            print("Overall Similarity:", overall_similarity)
            print("Answer3 Vehicle Info:", answer3_vehicle_info)
        else:
            answer1_description, overall_similarity = extract_answer1_overall_similarity(response)
            new_key = str(int(max(Scenario_database.keys(), key=int)) + 1) if Scenario_database else "0"
            Scenario_database[new_key] = answer1_description
            Scenario_database = OrderedDict(Scenario_database)
            answer3_vehicle_info = {}  # Leave answer3_vehicle_info empty
            print("Extracted Answer1 Description:", answer1_description)
            print("Extracted Overall Similarity:", overall_similarity)
            print("Answer3 Vehicle Info:", answer3_vehicle_info)

        ind.mutate_info = answer3_vehicle_info
        # reload scenario state
        ind.state = states.ScenarioState()
    except Exception as e:
        if e == TimeoutError:
            print("[-] simulation hanging. abort.")
            ret = 1
        else:
            print("[-] run_test error:")
            traceback.print_exc()
            exit(0)
    if ret is None:
        pass
    elif ret == -1:
        print("[-] Fatal error occurred during test")
        exit(0)
    elif ret == 1:
        print("fuzzer - found an error")
    elif ret == 128:
        print("Exit by user request")

    # mutation loop ends
    if ind.found_error:
        print("[-]error detected. start a new cycle with a new seed")
    return min_dist, nova, 100 - overall_similarity


# MUTATION OPERATOR


def mut_npc_list(ind: Scenario):
    global town_map
    if len(ind.npc_list) <= 1:
        return ind.npc_list
    if bottleneck:
        # mutate the chosen one by GPT
        for npc in ind.npc_list:
            if npc.instance_id == ind.mutate_info["Vehicle ID"]:
                npc.speed = ind.mutate_info["Speed"]
                if town_map is not None:
                    location = carla.Location(x=ind.mutate_info["Location"][0], y=ind.mutate_info["Location"][1],
                                              z=0.5)
                    waypoint = town_map.get_waypoint(location, project_to_road=True,
                                                     lane_type=carla.libcarla.LaneType.Driving)
                    npc.spawn_point = waypoint
                return ind.npc_list
        return ind.npc_list
    mut_pb = random.random()
    random_index = random.randint(0, len(ind.npc_list) - 1)
    # remove a random 1
    if mut_pb < 0.1:
        ind.npc_list.pop(random_index)
        return ind.npc_list
    # add a random 1
    if mut_pb < 0.4:
        template_npc = ind.npc_list[random_index]
        new_ad = NPC.get_npc_by_one(template_npc, town_map, len(ind.npc_list) - 1)
        ind.npc_list.append(new_ad)
        return ind.npc_list
    # mutate a random agent
    template_npc = ind.npc_list[random_index]
    new_ad = NPC.get_npc_by_one(template_npc, town_map, len(ind.npc_list) - 1)
    ind.npc_list.append(new_ad)
    ind.npc_list.pop(random_index)
    return ind.npc_list


def mut_scenario(ind: Scenario):
    ind.npc_list = mut_npc_list(ind)
    return ind,


# CROSSOVER OPERATOR

def cx_npc(ind1: List[NPC], ind2: List[NPC]):
    # todo: swap entire ad section
    cx_pb = random.random()
    if cx_pb < 0.05:
        return ind2, ind1

    for adc1 in ind1:
        for adc2 in ind2:
            NPC.npc_cross(adc1, adc2)

    # # if len(ind1.adcs) < MAX_ADC_COUNT:
    # #     for adc in ind2.adcs:
    # #         if ind1.has_conflict(adc) and ind1.add_agent(deepcopy(adc)):
    # #             # add an agent from parent 2 to parent 1 if there exists a conflict
    # #             ind1.adjust_time()
    # #             return ind1, ind2
    #
    # # if none of the above happened, no common adc, no conflict in either
    # # combine to make a new populations
    # available_adcs = ind1.adcs + ind2.adcs
    # random.shuffle(available_adcs)
    # split_index = random.randint(2, min(len(available_adcs), MAX_ADC_COUNT))
    #
    # result1 = ADSection([])
    # for x in available_adcs[:split_index]:
    #     result1.add_agent(copy.deepcopy(x))
    #
    # # make sure offspring adc count is valid
    #
    # while len(result1.adcs) > MAX_ADC_COUNT:
    #     result1.adcs.pop()
    #
    # while len(result1.adcs) < 2:
    #     new_ad = ADAgent.get_one()
    #     if result1.has_conflict(new_ad) and result1.add_agent(new_ad):
    #         break
    # result1.adjust_time()
    return ind1, ind2


def cx_scenario(ind1: Scenario, ind2: Scenario):
    ind1.npc_list, ind2.npc_list = cx_npc(
        ind1.npc_list, ind2.npc_list
    )
    return ind1, ind2


def seed_initialize(town, town_map):
    spawn_points = town.get_spawn_points()
    sp = random.choice(spawn_points)
    sp_x = sp.location.x
    sp_y = sp.location.y
    sp_z = sp.location.z
    pitch = sp.rotation.pitch
    yaw = sp.rotation.yaw
    roll = sp.rotation.roll
    # restrict destination to be within 200 meters
    destination_flag = True
    wp, wp_x, wp_y, wp_z, wp_yaw = None, None, None, None, None
    while destination_flag:
        wp = random.choice(spawn_points)
        wp_x = wp.location.x
        wp_y = wp.location.y
        wp_z = wp.location.z
        wp_yaw = wp.rotation.yaw
        if math.sqrt((sp_x - wp_x) ** 2 + (sp_y - wp_y) ** 2) > c.MIN_DIST:
            destination_flag = False
        if math.sqrt((sp_x - wp_x) ** 2 + (sp_y - wp_y) ** 2) > c.MAX_DIST:
            destination_flag = True
    seed_dict = {
        "map": town_map,
        "sp_x": sp_x,
        "sp_y": sp_y,
        "sp_z": sp_z,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
        "wp_x": wp_x,
        "wp_y": wp_y,
        "wp_z": wp_z,
        "wp_yaw": wp_yaw
    }
    return seed_dict


def init_env(args):
    conf = config.Config()

    if args is None:
        argument_parser = set_args()
        args = argument_parser.parse_args()

    ini_hyperparameters(conf, args)
    if conf.town is not None:
        town_map = "Town0{}".format(conf.town)
    else:
        town_map = "Town0{}".format(random.randint(1, 5))
    if conf.no_traffic_lights:
        conf.check_dict["red"] = False
    signal.signal(signal.SIGALRM, handler)
    client = utils.connect(conf)
    client.set_timeout(20)
    client.load_world(town_map)
    world = client.get_world()
    town = world.get_map()
    map_topology = town.get_topology()
    G = nx.DiGraph()
    lane_list = {}
    for edge in map_topology:
        # 1.add_edge for every lane that is connected
        G.add_edge((edge[0].road_id, edge[0].lane_id), (edge[1].road_id, edge[1].lane_id))
        if (edge[0].road_id, edge[0].lane_id) not in lane_list:
            edge_end = edge[0].next_until_lane_end(500)[-1]
            lane_list[(edge[0].road_id, edge[0].lane_id)] = (edge[0], edge_end)
    added_edges = []
    for lane_A in lane_list:
        for lane_B in lane_list:
            # 2.add_edge for every lane that is cross in junction
            if lane_A != lane_B:
                point_a = lane_list[lane_A][0].transform.location.x, lane_list[lane_A][0].transform.location.y
                point_b = lane_list[lane_A][1].transform.location.x, lane_list[lane_A][1].transform.location.y
                point_c = lane_list[lane_B][0].transform.location.x, lane_list[lane_B][0].transform.location.y
                point_d = lane_list[lane_B][1].transform.location.x, lane_list[lane_B][1].transform.location.y
                line_ab = LineString([point_a, point_b])
                line_cd = LineString([point_c, point_d])
                if line_ab.crosses(line_cd):
                    if (lane_B, lane_A) not in added_edges:
                        G.add_edge(lane_A, lane_B)
                        G.add_edge(lane_B, lane_A)
                        # added_edges.append((lane_A, lane_B))
    for lane in lane_list:
        # 3.add_edge for evert lane that could change to
        lane_change_left = lane_list[lane][0].lane_change == carla.LaneChange.Left or \
                           lane_list[lane][0].lane_change == carla.LaneChange.Both or \
                           lane_list[lane][1].lane_change == carla.LaneChange.Left or \
                           lane_list[lane][1].lane_change == carla.LaneChange.Both
        lane_change_right = lane_list[lane][0].lane_change == carla.LaneChange.Right or \
                            lane_list[lane][0].lane_change == carla.LaneChange.Both or \
                            lane_list[lane][1].lane_change == carla.LaneChange.Right or \
                            lane_list[lane][1].lane_change == carla.LaneChange.Both
        if lane_change_left:
            if (lane[0], lane[1] + 1) in lane_list:
                G.add_edge(lane, (lane[0], lane[1] + 1))
        if lane_change_right:
            if (lane[0], lane[1] - 1) in lane_list:
                G.add_edge(lane, (lane[0], lane[1] - 1))
    utils.switch_map(conf, town_map, client)
    return conf, town, town_map, client, world, G


def print_all_attr(obj):
    attributes = dir(obj)
    for attr_name in attributes:
        if not callable(getattr(obj, attr_name)):
            attr_value = getattr(obj, attr_name)
            attr_type = type(attr_value)
            print(f"Attribute: {attr_name}, Value: {attr_value}, Type: {attr_type}")


def check_nondominated_stability(pareto_front, archive, generations=10, epsilon=1e-6):
    """
    Checks stability of non-dominated solutions across multiple generations.
    pareto_front: The Pareto front (non-dominated solutions) of the current generation
    archive: A list storing historical Pareto fronts
    generations: Number of generations to compare
    epsilon: Threshold to determine if there's a significant change in the Pareto front
    """
    archive.append(pareto_front)

    if len(archive) < generations:
        return False  # Not enough generations yet for comparison

    # Compare differences over the recent generations
    for i in range(-2, -generations - 1, -1):
        prev_pareto = archive[i]
        if any(abs(x.fitness.values[0] - y.fitness.values[0]) > epsilon for x, y in zip(pareto_front, prev_pareto)):
            return False  # If the difference exceeds the threshold, there is no stagnation

    return True


def check_crowding_distance_stability(pareto_front, generations=10, epsilon=1e-6):
    """
    Checks stability of crowding distance.
    pareto_front: The Pareto front (non-dominated solutions) of the current generation
    generations: Number of generations to compare
    epsilon: Threshold to determine if there's a significant change in crowding distance
    """
    # Calculate the average crowding distance for the current generation's Pareto front
    crowding_distances = tools.sortNondominated(pareto_front, len(pareto_front))[0]
    avg_crowding_distance = np.mean([ind.fitness.crowding_dist for ind in crowding_distances])

    # If there are already `generations` number of records for crowding distance, compare them
    if len(crowding_distances) >= generations:
        recent_distances = [np.mean([ind.fitness.crowding_dist for ind in gen]) for gen in
                            crowding_distances[-generations:]]
        if all(abs(avg_crowding_distance - dist) < epsilon for dist in recent_distances):
            return True  # Small changes in crowding distance indicate possible stagnation

    return False


def check_objective_variance_stability(pareto_front, generations=10, epsilon=1e-6):
    """
    Checks stability of objective function variance.
    pareto_front: The Pareto front (non-dominated solutions) of the current generation
    generations: Number of generations to compare
    epsilon: Threshold to determine if there's a significant change in variance
    """
    # Calculate variance of the objectives in the current generation
    objectives = np.array([ind.fitness.values for ind in pareto_front])
    current_variance = np.var(objectives, axis=0)

    # If there are already `generations` number of records for variance, compare them
    if len(objectives) >= generations:
        recent_variances = [np.var([ind.fitness.values for ind in gen], axis=0) for gen in objectives[-generations:]]
        if all(np.all(np.abs(current_variance - var) < epsilon) for var in recent_variances):
            return True  # Small changes in objective function variance indicate possible stagnation

    return False


def check_diversity_bottleneck(current_pareto_front, archive, generations=10, epsilon=1e-6):
    checks = [
        check_nondominated_stability(current_pareto_front, archive, generations=generations, epsilon=epsilon),
        check_crowding_distance_stability(current_pareto_front, generations=generations, epsilon=epsilon),
        check_objective_variance_stability(current_pareto_front, generations=generations, epsilon=epsilon)
    ]

    satisfied_conditions = sum(checks)

    return satisfied_conditions >= 2


def main(args=None):
    # STEP 0: init env
    global client, world, G, blueprint_library, town_map, bottleneck
    logging.basicConfig(filename='./data/record.log', filemode='a', level=logging.INFO,
                        format='%(asctime)s - %(message)s')
    copyreg.pickle(carla.libcarla.Location, utils.carla_location_pickle, utils.carla_location_unpickle)
    copyreg.pickle(carla.libcarla.Rotation, utils.carla_rotation_pickle, utils.carla_rotation_unpickle)
    copyreg.pickle(carla.libcarla.Transform, utils.carla_transform_pickle, utils.carla_transform_unpickle)
    # copyreg.pickle(carla.libcarla.ActorBlueprint, carla_ActorBlueprint_pickle, carla_ActorBlueprint_unpickle)

    conf, town, town_map, exec_state.client, exec_state.world, exec_state.G = init_env(args)
    world = exec_state.world
    blueprint_library = world.get_blueprint_library()
    # if conf.agent_type == c.AUTOWARE:
    #     autoware_launch(exec_state.world, conf, town)
    population = []
    archive = []
    # GA Hyperparameters
    POP_SIZE = 5  # amount of population
    OFF_SIZE = 5  # number of offspring to produce
    MAX_GEN = 5  #
    CXPB = 0.8  # crossover probability
    MUTPB = 0.2  # mutation probability
    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluation)
    toolbox.register("mate", cx_scenario)
    toolbox.register("mutate", mut_scenario)
    toolbox.register("select", tools.selNSGA2)
    hof = tools.ParetoFront()
    # Evaluate Initial Population
    print(f' ====== Analyzing Initial Population ====== ')
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("max", np.max, axis=0)
    stats.register("min", np.min, axis=0)
    logbook = tools.Logbook()
    logbook.header = 'gen', 'avg', 'max', 'min'
    # begin a generational process
    curr_gen = 0
    # init some seed if seed pool is empty
    for i in range(POP_SIZE):
        seed_dict = seed_initialize(town, town_map)
        # Creates and initializes a Scenario instance based on the metadata
        with concurrent.futures.ThreadPoolExecutor() as my_simulate:
            future = my_simulate.submit(create_test_scenario, conf, seed_dict)
            test_scenario = future.result(timeout=15)
        population.append(test_scenario)
        test_scenario.scenario_id = len(population)
    while True:
        # Main loop
        curr_gen += 1
        if curr_gen > MAX_GEN:
            break
        print(f' ====== GA Generation {curr_gen} ====== ')
        # Vary the population
        offspring = algorithms.varOr(
            population, toolbox, OFF_SIZE, CXPB, MUTPB)
        # update chromosome generation_id and scenario_id
        for index, d in enumerate(offspring):
            d.generation_id = curr_gen
            d.scenario_id = index
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        current_pareto_front = [ind for ind in population if ind in hof.items]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        hof.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, POP_SIZE)
        record = stats.compile(population)

        # Combined Bottleneck Detection
        if check_diversity_bottleneck(current_pareto_front, archive, generations=10, epsilon=1e-6):
            print("GA has entered a bottleneck, stopping evolution or taking alternative actions")
            bottleneck = True
        else:
            bottleneck = False
        logbook.record(gen=curr_gen, **record)
        print(logbook.stream)
        # Save directory for trace graphs


if __name__ == "__main__":
    main()
