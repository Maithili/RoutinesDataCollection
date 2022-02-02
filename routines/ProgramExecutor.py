import json
import sys
sys.path.append('..')
from evolving_graph.scripts import Script, parse_script_line
from evolving_graph.environment import EnvironmentGraph
from evolving_graph.execution import ScriptExecutor
import evolving_graph.utils as utils
from object_locations import object_locations

def class_from_id(graph, id):
    lis = [n['class_name'] for n in graph['nodes'] if n['id']==id]
    if len(lis) > 0:
        return lis[0]
    else:
        return 'None'

def print_graph_difference(g1,g2):
    edges_removed = [e for e in g1['edges'] if e not in g2['edges']]
    edges_added = [e for e in g2['edges'] if e not in g1['edges']]
    nodes_removed = [n for n in g1['nodes'] if n['id'] not in [n2['id'] for n2 in g2['nodes']]]
    nodes_added = [n for n in g2['nodes'] if n['id'] not in [n2['id'] for n2 in g1['nodes']]]
    ignore_for_edges = ['wall']

    for n in nodes_removed:
        print ('Removed node : ',n)
    for n in nodes_added:
        print ('Added node   : ',n)
    remaining_objects = []
    for e in edges_removed:
        c1 = class_from_id(g1,e['from_id'])
        c2 = class_from_id(g1,e['to_id'])
        if c1 not in ignore_for_edges and c2 not in ignore_for_edges and e['relation_type'] in ['INSIDE','ON','HOLDS_RH','HOLDS_LH']:
            print (' - ',c1,e['relation_type'],c2)
            remaining_objects.append(e['from_id'])
    for e in edges_added:
        c1 = class_from_id(g2,e['from_id'])
        c2 = class_from_id(g2,e['to_id'])
        if c1 not in ignore_for_edges and c2 not in ignore_for_edges and e['relation_type'] in ['INSIDE','ON','HOLDS_RH','HOLDS_LH']:
            print (' + ',c1,e['relation_type'],c2)
            if e['from_id'] in remaining_objects:
                remaining_objects.remove(e['from_id'])
    # for id in remaining_objects:
    #     for e in g2['edges']:
    #         if e['from_id'] == id and e['relation_type'] in ['INSIDE','ON']:
    #             c2 = class_from_id(g2,e['to_id'])
    #             if c2 not in ignore_for_edges:
    #                 c1 = class_from_id(g2,e['from_id'])
    #                 print (' + ',c1,e['relation_type'],c2)


def read_program(file_name, node_map):
    action_headers = []
    action_scripts = []
    action_objects_in_use = []

    def obj_class_id_from_string(string_in):
        class_id = [a[1:-1] for a in string_in.split(' ')]
        return (int(class_id[1]), class_id[0])

    with open(file_name) as f:
        lines = []
        full_program = []
        obj_start, obj_end = [], []
        index = 1
        object_use = {'start':[], 'end':[]}
        for line in f:
            if line.startswith('##'):
                header = line[2:].strip()
                action_headers.append(header)
                action_scripts.append(lines)
                object_use['start'].append(obj_start)
                object_use['end'].append(obj_end)
                lines = []
                obj_start, obj_end = [], []
                index = 1
            line = line.strip()
            if line.startswith('+'):
                obj = obj_class_id_from_string(node_map[line[1:]])
                obj_start.append(obj)
                continue
            if line.startswith('-'):
                obj = obj_class_id_from_string(node_map[line[1:]])
                obj_end.append(obj)
                continue
            if '[' not in line:
                continue
            if len(line) > 0 and not line.startswith('#'):
                mapped_line = line
                for full_name, name_id in node_map.items():
                    mapped_line = mapped_line.replace(full_name, name_id)
                # print(line,' -> ', mapped_line)
                try:
                    scr_line = parse_script_line(mapped_line, index, custom_patt_params = r'\<(.+?)\>\s*\((.+?)\)')
                except Exception as e:
                    print(f'The following line has a mistake! Did you write the correct object and activity names? \n {line}')
                    raise e
                lines.append(scr_line)
                full_program.append(scr_line)
                index += 1
        action_scripts.append(lines)
        action_scripts = action_scripts[1:]
    return action_headers, action_scripts, object_use, full_program

def execute_program(program_file, graph_file, node_map, verbose=False):
    with open (graph_file,'r') as f:
        init_graph_dict = json.load(f)
    init_graph = EnvironmentGraph(init_graph_dict)
    action_headers, action_scripts, action_obj_use, whole_program = read_program(program_file, node_map)
    name_equivalence = utils.load_name_equivalence()
    graphs = [init_graph.to_dict()]

    print('Checking scripts...',end='')
    save_graph = [0]
    for a in action_scripts:
        save_graph.append(save_graph[-1] + len(a))
    executor = ScriptExecutor(EnvironmentGraph(graphs[-1]), name_equivalence)
    success, state, graph_list = executor.execute(Script(whole_program), w_graph_list=True)
    # graph_list = [init_graph_dict] + graph_list[1:]
    print('exec info ---- ')
    print(executor.info.get_error_string())
    if not success:
        error_str = executor.info.get_error_string()
        if 'inside other closed thing' in error_str:
            object = error_str[error_str.index('<')+1:error_str.index('>')]
            print(f'{object} is inside {object_locations[object]}')
        raise RuntimeError(f'Execution failed because {error_str}')
    print('Execution successful!!')

    print('Checking final state...',end='')
    executor = ScriptExecutor(EnvironmentGraph(state.to_dict()), name_equivalence)
    executor.check_final_state()
    print('Final state OK\n')

    assert len(save_graph)-1 == len(action_scripts)
    if verbose:
        print('This is how the scene changes after every set of actions...')
        for idx, script in zip(save_graph[1:],action_scripts):
            print('\nScript : ')
            for l in script:
                print('  - ',l)
            graphs.append(graph_list[idx])
            print('Changes : ')
            if verbose:
                print_graph_difference(graphs[-2],graphs[-1])
    