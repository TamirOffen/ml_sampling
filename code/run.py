import argparse
from MapEnvironment import MapEnvironment
from RRTMotionPlanner import RRTMotionPlanner
import numpy as np
from AdaptiveSampler2D import AdaptiveSampler2D
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='script for testing planners')
    parser.add_argument('-map', '--map', type=str, default='map_mp.json', help='Json file name containing all map information')
    # parser.add_argument('-task', '--task', type=str, default='mp', help='choose from mp (motion planning) and ip (inspection planning)')
    # parser.add_argument('-ext_mode', '--ext_mode', type=str, default='E1', help='edge extension mode')
    # parser.add_argument('-goal_prob', '--goal_prob', type=float, default=0.05, help='probability to draw goal vertex')
    # parser.add_argument('-coverage', '--coverage', type=float, default=0.5, help='percentage of points to inspect (inspection planning)')
    args = parser.parse_args()

    # prepare the map
    # planning_env = MapEnvironment(json_file=args.map, origin='center')

    # setup the planner
    # if args.task == 'mp':
    #     planner = RRTMotionPlanner(planning_env=planning_env, ext_mode=args.ext_mode, goal_prob=args.goal_prob)
    # elif args.task == 'ip':
    #     planner = RRTInspectionPlanner(planning_env=planning_env, ext_mode=args.ext_mode, goal_prob=args.goal_prob, coverage=args.coverage)
    # else:
    #     raise ValueError('Unknown task option: %s' % args.task)

    # execute plan
    # plan = planner.plan()
    

    # Visualize the final path.
    # planner.planning_env.visualize_plan(plan)


    map_env = MapEnvironment(json_file="map_mp.json", origin='center')
    print("drawing config space")
    map_env.draw_config_space(resolution=0.025)
    print("drawing sampled config space")
    map_env.draw_sampled_config_space(iterations=1000, resolution=0.05)
    print("drawing uniform sampled config space")
    map_env.draw_sampled_config_space(iterations=1000, resolution=0.05, uniform=True)
    print("planning a new place")
    planner = RRTMotionPlanner(planning_env=map_env, ext_mode="E2", goal_prob=0.05)

    plan = planner.plan()
    print(plan)

    plan = np.array([[1.8, -1.5], [1.8, -4]])
    print(plan)

    planner.planning_env.visualize_plan(plan)

    #sampler = AdaptiveSampler2D(legal_config_func=map_env.config_validity_checker)
    # print(sampler.pdf)
    #sampler.run(num_iterations=20)
