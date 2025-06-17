import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import random
from copy import deepcopy
from itertools import product
import pulp as pu
import re
import seaborn as sns
import colorcet as cc
import matplotlib.colors as mcolors
import math
from PolynomialClass import *


def process_expression(expr, graph, node_id, parent_id=None, top_level=False, expr_to_id=None):

    
    if expr_to_id is None:
        expr_to_id = {}
    
    # Checks if the exact expression has been processed and does not add it if so
    # Uses hash to create a unique key for each expression element
    expr_hash = hash(expr)
    if expr_hash in expr_to_id:
        node_for_expr = expr_to_id[expr_hash]
        if parent_id is not None:
            graph.add_edge(node_for_expr, parent_id)

        return node_id, node_for_expr
    
    #  Addition and Multiplication operations
    if type(expr) in (Addition, Multiplication):
        args = list(expr.args)
        
        if top_level:
            # if top_level = True the first time this function executes, there is a non binary addition as the last addition in the sequence of polynomials
            add_node = node_id
            graph.add_node(add_node, label="+" if type(expr) == Addition else "*", type="operation")
            expr_to_id[expr_hash] = add_node
            node_id += 1
            
            # Process each term in the top-level expression
            for term in args:
                node_id, child_id = process_expression(term, graph, node_id, add_node, top_level=False, expr_to_id=expr_to_id)
            
            if parent_id is not None:
                graph.add_edge(add_node, parent_id)
            return node_id, add_node
        else:
            # Creates a binary tree for addition/multiplication
            if type(expr) == Addition:
                label = "+"
            else:
                label = "*"
                
            if len(args) > 0:
                left_expr = args[0]
                node_id, left_id = process_expression(left_expr, graph, node_id, top_level=False, expr_to_id=expr_to_id)
        
                for i in range(1, len(args)):
                    expr_branch = args[i]
                    parent_node = node_id
                    graph.add_node(parent_node, label=label, type="operation")
                    expr_to_id[expr_hash] = parent_node
                    node_id += 1
                    
                    # Connect to the parent node if exists
                    if parent_id is not None:
                        graph.add_edge(parent_node, parent_id)
                        parent_id = parent_node
                    
                    # Left branch is the result so far
                    graph.add_edge(left_id, parent_node, label='left')
                    
                    # Right branch is the current term
                    node_id, right_id = process_expression(expr_branch, graph, node_id, top_level=False, expr_to_id=expr_to_id)
                    graph.add_edge(right_id, parent_node, label='right')
                    
                    left_id = parent_node 
                
                return node_id, left_id


    # Power operation
    elif type(expr) == Power:
        base, exp = expr.args
        if hasattr(exp, 'is_integer') and exp.is_integer and exp > 1:
            return convert_power_to_multiplication(expr, base, exp, graph, node_id, parent_id, expr_to_id)

            
    elif type(expr) == Variable or hasattr(expr, 'is_symbol') and expr.is_symbol():
        curr_id = node_id
        graph.add_node(curr_id, label=str(expr), type="value")
        expr_to_id[expr_hash] = curr_id
        node_id += 1
        if parent_id is not None:
            graph.add_edge(curr_id, parent_id)
        return node_id, curr_id
    
    elif hasattr(expr, 'is_number') and expr.is_number():
        curr_id = node_id
        # Negative numbers directly with the negative sign
        graph.add_node(curr_id, label=str(expr), type="value")
        expr_to_id[expr_hash] = curr_id
        node_id += 1
        if parent_id is not None:
            graph.add_edge(curr_id, parent_id)
        return node_id, curr_id
    
    elif type(expr) == UnaryMinus:

        mult_node = node_id
        graph.add_node(mult_node, label="*", type="operation")
        expr_to_id[expr_hash] = mult_node
        node_id += 1
        
        if parent_id is not None:
            graph.add_edge(mult_node, parent_id)
        
        # -1 node
        const_node = node_id
        graph.add_node(const_node, label="-1", type="value")
        node_id += 1
        graph.add_edge(const_node, mult_node, label='left')
        

        node_id, term_id = process_expression(expr.value, graph, node_id, mult_node, top_level=False, expr_to_id=expr_to_id)
        graph.add_edge(term_id, mult_node, label='right')
        
        return node_id, mult_node
    
    elif type(expr) == Subtraction:
        sub_node = node_id
        graph.add_node(sub_node, label="-", type="operation")
        expr_to_id[expr_hash] = sub_node
        node_id += 1
        
        if parent_id is not None:
            graph.add_edge(sub_node, parent_id)
        
        node_id, left_id = process_expression(expr.left, graph, node_id, sub_node, top_level=False, expr_to_id=expr_to_id)
        graph.add_edge(left_id, sub_node, label='left')
        
        node_id, right_id = process_expression(expr.right, graph, node_id, sub_node, top_level=False, expr_to_id=expr_to_id)
        graph.add_edge(right_id, sub_node, label='right')
        
        return node_id, sub_node




def convert_power_to_multiplication(expr, base, exp, graph, node_id, parent_id, expr_to_id):
    expr_hash = hash(expr)
    if expr_hash in expr_to_id:
        existing_node_id = expr_to_id[expr_hash]
        if parent_id is not None:
            graph.add_edge(existing_node_id, parent_id)
        return node_id, existing_node_id

    if exp == 1:
        return process_expression(base, graph, node_id, parent_id, top_level=False, expr_to_id=expr_to_id)

    if exp == 2:
        node_id, base_id = process_expression(base, graph, node_id, top_level=False, expr_to_id=expr_to_id)
        mul_node = node_id
        graph.add_node(mul_node, label="*", type="operation")
        expr_to_id[expr_hash] = mul_node
        node_id += 1

        graph.add_edge(base_id, mul_node, label='left')
        graph.add_edge(base_id, mul_node, label='right')

        if parent_id is not None:
            graph.add_edge(mul_node, parent_id)
        return node_id, mul_node

    # Calculates the exponents by splitting into two even branches
    if exp % 2 == 0:
        left_exp = right_exp = exp // 2
    else:
        left_exp = 1
        right_exp = exp - 1

    left_expr = base**left_exp
    if hash(left_expr) in expr_to_id:
        left_id = expr_to_id[hash(left_expr)]
    else:
        node_id, left_id = convert_power_to_multiplication(left_expr, base, left_exp, graph, node_id, None, expr_to_id)

    right_expr = base**right_exp
    if hash(right_expr) in expr_to_id:
        right_id = expr_to_id[hash(right_expr)]
    else:
        node_id, right_id = convert_power_to_multiplication(right_expr, base, right_exp, graph, node_id, None, expr_to_id)

    # Creates multiplication node
    mul_node = node_id
    graph.add_node(mul_node, label="*", type="operation")
    expr_to_id[expr_hash] = mul_node
    node_id += 1

    graph.add_edge(left_id, mul_node, label='left')
    graph.add_edge(right_id, mul_node, label='right')

    if parent_id is not None:
        graph.add_edge(mul_node, parent_id)

    return node_id, mul_node



class GenerateTree:

    def __init__(self, graph, polynomial_str, specific_node_colors = None, provided_levels = None, display = True):
        self.display_graph(graph, polynomial_str, specific_node_colors, provided_levels, display)


    def display_graph(self, graph, polynomial_str, specific_node_colors = None, provided_levels = None, display = True):
        
        if specific_node_colors is None:
            specific_node_colors = {}

        node_labels = nx.get_node_attributes(graph, 'label')
        if provided_levels is None:
            levels = self.stage_levels(graph)
            pos = self.custom_layout(levels)
        else:
            levels = self.update_existing_levels(graph, provided_levels)
            # print("Levels here: ", levels)
            new_levels = levels
            pos = self.custom_layout(new_levels)

            

        if display:
            plt.figure(figsize=(17, 35))
            
            operation_nodes = [n for n in graph.nodes() if graph.nodes[n]['type'] == 'operation']
            value_nodes = [n for n in graph.nodes() if graph.nodes[n]['type'] == 'value']
            shift_nodes = [n for n in graph.nodes() if graph.nodes[n]['type'] == 'shift']
            register_nodes = [n for n in graph.nodes() if graph.nodes[n]['type'] == 'register']
            pshift_nodes = [n for n in graph.nodes() if graph.nodes[n]['type'] == 'pshift']

            operation_node_colors = [specific_node_colors[node] if node in specific_node_colors else 'violet'
                                    for node in operation_nodes]


            edges = nx.draw_networkx_edges(graph, pos, arrows=True, arrowstyle='->', arrowsize=20, width=1.5, edge_color='gray')
            for edge in edges:
                edge.set_zorder(2)
            
            nodes = nx.draw_networkx_nodes(graph, pos, nodelist=operation_nodes, node_size=1800, node_color = operation_node_colors, alpha=0.9)
            nodes.set_zorder(1)

            nodes = nx.draw_networkx_nodes(graph, pos, nodelist=value_nodes, node_size=1500, node_color='paleturquoise', alpha=0.9)
            nodes = nx.draw_networkx_nodes(graph, pos, nodelist=shift_nodes, node_size=1500, node_color='violet', alpha=0.9)
            nodes = nx.draw_networkx_nodes(graph, pos, nodelist=register_nodes, node_size=1500, node_color='gold', alpha=0.9)
            nodes = nx.draw_networkx_nodes(graph, pos, nodelist=pshift_nodes, node_size=1500, node_color='gold', alpha=0.9)


            nodes.set_zorder(1)

            edge_labels = nx.get_edge_attributes(graph, 'label')

        # Draw edge labels
            drawn_labels = nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

            for text in drawn_labels.values():
                text.set_zorder(3)  
                   
            labels = {}
            for node in graph.nodes():
                attr_value = graph.nodes[node]['label']
                
                if check_numeric(attr_value):
                    formatted_value = f"{float(attr_value):.3f}".rstrip('0').rstrip('.')
                else:
                    formatted_value = attr_value
                
                # labels[node] = f"{node}: {formatted_value}"
                labels[node] = f"{formatted_value}"
            
            # graph_labels = nx.draw_networkx_labels(graph, pos, labels=labels, font_size=17, font_weight='bold')
            
            operation_labels = {node: labels[node] for node in graph.nodes() if graph.nodes[node]['label'] in ['+', '-', '*']}
            other_labels = {node: labels[node] for node in graph.nodes() if graph.nodes[node]['label'] not in ['+', '-', '*']}

            if operation_labels:
                graph_labels_ops = nx.draw_networkx_labels(graph, pos, labels=operation_labels, font_size=22, font_weight='bold')
                for text in graph_labels_ops.values():
                    text.set_zorder(3)

            if other_labels:
                graph_labels_others = nx.draw_networkx_labels(graph, pos, labels=other_labels, font_size=15, font_weight='bold')
                for text in graph_labels_others.values():
                    text.set_zorder(3)
            

            # plt.title(f"Data Flow Graph of: ${(polynomial_str.replace('**', '^')).replace('*', '')}$", fontsize=16)
            plt.title('Data Flow Graph')

            plt.subplots_adjust()
            plt.subplots_adjust(left=0.01, right=0.99, top=1.05, bottom=0.01)

            plt.axis('off')
            # plt.tight_layout()
            # plt.savefig("Non_factorised_ConstrainedSR3DFG.pdf", bbox_inches='tight', pad_inches=0)
            plt.show() 


    def update_existing_levels(self, G, levels):

        all_nodes = list(nx.topological_sort(G))

        labelled_nodes = []

        for u, v in levels.items():
            labelled_nodes.append(u)

        new_nodes = list(set(all_nodes) - set(labelled_nodes))
        # print("New nodes are: ", new_nodes)

        expanded_levels = levels
        for node in labelled_nodes:
            parent = list(G.predecessors(node))

            for j in new_nodes:
                if j in parent:
                    expanded_levels[j] = expanded_levels[node] - 1

        # print(expanded_levels)

        return expanded_levels


    def stage_levels(self, G):
        levels = {}
        topo_order = list(nx.topological_sort(G))
        for node in topo_order:
            preds = list(G.predecessors(node))
            if not preds:
                levels[node] = 0
            else:

                levels[node] = max(levels[p] + 1 for p in preds)

        
        for node in topo_order:
            preds = list(G.predecessors(node))
            if preds:
                labels = [G.nodes[s]['label'] for s in preds]
                shift_count = sum('<<' in label or '>>' in label for label in labels)
                pred_diff = max([levels[node] - levels[s] for s in preds])
                
                # Differences in levels between node and its predecessors
                # Needs a check because node can only be moved relative to the bit shift if its other predecessor allows it
                can_shift = True
                
                for s in preds:
                    if levels[node] - 1 <= levels[s]:
                        can_shift = False
                        break

                if can_shift and (shift_count == 2 or (shift_count == 1 and all('<<' in label or '>>' in label or label.isalnum() for label in labels)) or (shift_count == 1 and pred_diff > 1)):
                    levels[node] = levels[node] - 1

        return levels

    def organise_nodes_by_level(self, levels):
        nodes_by_level = {}
        
        for node, level in levels.items():
            if level not in nodes_by_level:
                nodes_by_level[level] = []
            nodes_by_level[level].append(node)

        return {k: v for k, v in sorted(nodes_by_level.items())}



    def custom_layout(self, levels):
        pos = {}
        nodes_by_level = self.organise_nodes_by_level(levels)
        
        vertical_spacing = 3
        
        for level, nodes in nodes_by_level.items():
            num_nodes = len(nodes)
            y = (max(nodes_by_level.keys()) - level) * vertical_spacing
            
            if num_nodes == 1:

                x = random.uniform(-0.3, 0.3)
                pos[nodes[0]] = (x, y)
            else:

                total_width = min(8.0, num_nodes * 1.0) 
                x_spacing = total_width / (num_nodes - 1) if num_nodes > 1 else 0
                x_start = -total_width / 2
                
                for i, node in enumerate(sorted(nodes)):
                    x = x_start + i * x_spacing
                    x += random.uniform(-0.1, 0.1)
                    pos[node] = (x, y)
        
        return pos
        
    

class DSPSolver:
    def __init__(self, graph, node_levels):
        self.original_graph = graph
        self.node_levels = node_levels
        self.dsp_combos = self.generate_valid_combinations()
        self.graph, self.BitShift_graph = self.remove_non_dsp_nodes(graph)
        

    def generate_valid_combinations(self):
        operations = ['+', '-', '*']
        combos_by_length = []

        for length in range(1, 4):
            for combo in product(operations, repeat=length):
                combo = tuple(combo)

                if length == 1:
                    combos_by_length.append(combo)

                elif length == 2:
                    if combo.count('*') <= 1:
                        combos_by_length.append(combo)

                elif length == 3:
                    if combo[1] == '*' and combo.count('*') == 1:
                        combos_by_length.append(combo)

        return combos_by_length


    def remove_non_dsp_nodes(self, G):
        G_new = deepcopy(G)
        remove_nodes = []

        for node in G_new.nodes():
            label = G_new.nodes[node]['label']
            node_type = G_new.nodes[node]['type']
            if label.isalnum() or "<<" in label or ">>" in label:
                remove_nodes.append(node)

            if check_numeric(label) and node_type == "value":
                remove_nodes.append(node)

        for node in remove_nodes:
            G_new.remove_node(node)

        #GenerateTree(G_new, "Without bit shifts")

        return G_new, G


    def graph_operation_combos(self, G, G_with_bitshifts, dsp_combos, node_levels):
        possible_combos = []

        for node in G.nodes():
            val = G.nodes[node]['label']
            if (val,) in dsp_combos:
                possible_combos.append([node])

        # Multi node patterns
        for node in G.nodes():
            visited_nodes = []
            queue = deque([(node, [node])])

            while queue:
                current, path = queue.popleft()

                if tuple(path) in visited_nodes:
                    continue
                visited_nodes.append(tuple(path))

                if len(path) > 1:
                    sub_graph = G.subgraph(path)

                    # Check that graph is connected
                    if not nx.is_weakly_connected(sub_graph):
                        continue

                    operations = tuple([G.nodes[n]['label'] for n in path])

                    if operations in dsp_combos:
                        should_add = True
                        
                        for m in range(len(operations)):
                            if (operations[m] == '-' or operations[m] == '+') and m > 0:
                            
                                edges = list(G.in_edges(path[m], data=True))
                                combo_edge = None
                                
                                for edge in edges:

                                    # Edge comes from previous node in combo
                                    if edge[0] == path[m-1]:  
                                        combo_edge = edge
                                        break
                                
                                if combo_edge is None or combo_edge[2]['label'] != 'right':
                                    should_add = False
                                    break
                        
                        if should_add:
                            possible_combos.append(path)

                if len(path) > 3:
                    continue

                neighbours = list(G.successors(current))
                for k in neighbours:
                    if k not in path:
                        queue.append((k, path + [k]))

        # Checks duplicates
        unique_possible_combos = []
        seen = []

        for j in possible_combos:
            if j not in seen:
                seen.append(j)
                unique_possible_combos.append(j)


        
        # print("List: ", unique_possible_combos)

        final_combos = []

        for combo in unique_possible_combos:
            if len(combo) == 1:
                final_combos.append(combo)
                continue
            
            # Two nodes in a DSP block cannot have a bit shift between them
            valid_combo = True
            for node in combo[1:]:
                two_nodes_ago = []
                parent = list(G_with_bitshifts.predecessors(node))
                
                for w in parent:
                    grandparent = list(G_with_bitshifts.predecessors(w))
                    two_nodes_ago.extend(grandparent)
                
                for grandparent in two_nodes_ago:

                    if G_with_bitshifts.nodes[grandparent]['type'] != 'operation':
                        continue

                    if node in list(G_with_bitshifts.successors(grandparent)):
                        valid_combo = False
                        break
                
                if not valid_combo:
                    break
            
            # Non-last node in dsp combo cannot have more than one child
            if valid_combo:
                for node in combo[:-1]:
                    if len(list(G.successors(node))) > 1:
                        valid_combo = False
                        break
            
            # DSP Node cannot square the result of the pre-adder
            # i.e.: Combo is not valid if the first node is an addition
            # and the second is a multiplier and only has one edge coming in

            if valid_combo:
                if G.nodes[combo[0]]['label'] == '+':
                    edges_in = list(G.in_edges(combo[1]))
                    if G.nodes[combo[1]]['label'] == '*':
                        valid_combo = False

            if valid_combo:
                final_combos.append(combo)

        return final_combos


    def disjoint(self, list1, list2):
        return not any(item in list2 for item in list1)

    # Implementation of the greedy algorithm
    def GreedySolver(self):
        specific_combos = self.graph_operation_combos(self.graph, self.BitShift_graph, self.dsp_combos, self.node_levels)
        universe = set(self.graph.nodes())
        covered = set()
        solutions = []

        while universe:
            best_combo = None
            most_nodes = 0

            for combo in specific_combos:
                if not self.disjoint(combo, covered):
                    continue

                if len(combo) > most_nodes:
                    best_combo = combo
                    most_nodes = len(combo)


            solutions.append(best_combo)
            covered.update(best_combo)
            universe -= set(best_combo)

        return solutions

    # Solving is defined as a property of the class
    @property
    def Solver(self):
        specific_combos = self.graph_operation_combos(self.graph, self.BitShift_graph, self.dsp_combos, self.node_levels)

        # print("Possible combos: ", specific_combos)
        problem = pu.LpProblem("DSPSolver", sense=pu.LpMinimize)
        binary_dsp_combos = {}

        for i, combo in enumerate(specific_combos):
            binary_dsp_combos[i] = pu.LpVariable(f"combo_{i}", cat = 'Binary')

        # Goal
        problem += pu.lpSum(binary_dsp_combos.values())

        # Constraint
        for node in self.graph.nodes():

            combos_with_node = []
            for i, combo in enumerate(specific_combos):
                if node in combo:
                    combos_with_node.append(i)

            problem += pu.lpSum(binary_dsp_combos[k] for k in combos_with_node) == 1

        solver = pu.PULP_CBC_CMD(timeLimit=30, msg = False)
        # Stops after finding one solution
        result = problem.solve(solver)

        if pu.LpStatus[result] != 'Optimal':
            print("No optimal")
            return self.GreedySolver()
        else:
            print("OPTIMAL FOUND ILP")
        
        
        ilp_solutions = []
        for j, combo in binary_dsp_combos.items():
            if combo.value() == 1:
                    ilp_solutions.append(specific_combos[j])

        return ilp_solutions



def check_numeric(s):
    s = s.strip()
    try:
        float(s)
        return True
    except ValueError:
        return False


def merge_negatives(G):
    G_mod = deepcopy(G)
    nodes_to_remove = []

    new_node = []
    new_edges = []

    for node in list(G_mod.nodes()):
        if G_mod.nodes[node]['label'] == '*':
            parent_edges = list(G_mod.in_edges(node, data=True))
            
            if len(parent_edges) > 1:
                parent1, parent2 = parent_edges[0][0], parent_edges[1][0]
                
                if (G_mod.nodes[parent1]['type']== 'value' and 
                    G_mod.nodes[parent2]['type'] == 'value' and
                    check_numeric(G_mod.nodes[parent1]['label']) and 
                    check_numeric(G_mod.nodes[parent2]['label'])):
                    
                    # Calculate new value
                    new_value = (float(G_mod.nodes[parent1]['label']) * float(G_mod.nodes[parent2]['label']))
                    
                    children_edges = list(G_mod.out_edges(node, data=True))
                    nodes_to_remove.append(node)
                    nodes_to_remove.append(parent1)
                    nodes_to_remove.append(parent2)


                    new_node.append(((node, {'label': str(new_value)})))
                    new_edges.append(children_edges[0])


    for i in nodes_to_remove:
        # Remove all edges connected to the node
        G_mod.remove_edges_from(list(G.in_edges(i))) 
        G_mod.remove_edges_from(list(G.out_edges(i)))
        
        # Remove the node
        G_mod.remove_node(i)

    for i in new_node:
        G_mod.add_node(i[0], label= i[1]['label'], type="value")

    for j in new_edges:
        G_mod.add_edge(j[0], j[1], label = j[2]['label'])


    G_relabelled = relabel_graph(G_mod)
        

    return G_relabelled


def relabel_graph(G):
    old_nodes = sorted(G.nodes())
    new_labels = {old: new for new, old in enumerate(old_nodes)}

    G_relabelled = nx.relabel_nodes(G, new_labels, copy=True)

    return G_relabelled



def implement_bit_shifts(G):
    remove_vals = [] 
    # Contains the multiplication node. Remove the nodes and its edges
    remove_nodes = []  
    # Each sub list has structure [(.. Bit shifts .., operation), parent node, child node, child edge label]
    bit_shift_nodes = [] 
    single_bit_shift_nodes = []
    
    for node in G.nodes():
        if G.nodes[node]['label'] == '*' and len(G.in_edges(node)) == 2:

            edges = list(G.in_edges(node, data=True))
            
            left_edge = None
            right_edge = None
            for tup in edges:
                if 'label' in tup[2] and tup[2]['label'] == 'left':
                    left_edge = tup
                elif 'label' in tup[2] and tup[2]['label'] == 'right':
                    right_edge = tup
            
            if left_edge is None or right_edge is None:
                continue

            # If left edge is not a number then continue with the loop
            if not check_numeric(G.nodes[left_edge[0]]['label']):
                continue

            # Skip multiplication nodes with -1
            if G.nodes[left_edge[0]]['label'] == '-1':
                continue
            
            val = G.nodes[left_edge[0]]['label']
            constant_node = left_edge[0]

            # Remove constants that are exclusively used by multiplication nodes
            constant_out_edges = list(G.out_edges(constant_node))
            all_targets_are_multiplication = True
            
            for _, target in constant_out_edges:
                if target in G.nodes() and G.nodes[target]['label'] != '*':
                    all_targets_are_multiplication = False
                    break
            
            if all_targets_are_multiplication:
                remove_vals.append(constant_node)

            remove_nodes.append(node) 
            parent_node = right_edge[0]
            
         
            child_connections = []
            child_edges = list(G.out_edges(node, data=True))
            for edge in child_edges:
                child_node = edge[1]
                edge_data = edge[2]
                child_edge_label = edge_data['label']
                child_connections.append((child_node, child_edge_label))

            # If the multiplication has no children it is at the bottom
            if not child_connections:
                child_connections.append((None, None))

            if float(val) > 0:
                if math.log2(float(val)).is_integer():
                    shift_val = math.log2(float(val))

                    for child_node, child_edge_label in child_connections:
                        single_bit_shift_nodes.append([shift_val, parent_node, child_node, child_edge_label])
                else:
                    best_combination, best_sum = closest_power_of_2(float(val))

                    for child_node, child_edge_label in child_connections:
                        bit_shift_nodes.append([best_combination, parent_node, child_node, child_edge_label])

    # Runs if it is a sum combination of bit shift nodes
    if single_bit_shift_nodes:
        max_id = max(G.nodes()) + 1
        
        for k, series in enumerate(single_bit_shift_nodes):
            shift_val, parent_node, child_node, child_edge_label = series

            if shift_val >= 0:
                label = "<< " + str(int(shift_val))
            elif shift_val < 0:
                label = ">> " + str(int(-shift_val))

            G.add_node(max_id, label=label, type="shift")
            G.add_edge(parent_node, max_id)
            
            if child_node is not None:
                if child_edge_label is not None:
                    G.add_edge(max_id, child_node, label=child_edge_label)
                else:
                    G.add_edge(max_id, child_node)

            max_id += 1
    
    if bit_shift_nodes:
        max_id = max(G.nodes()) + 1
        for j, combo in enumerate(bit_shift_nodes):
            best_combination, parent_node, child_node, child_edge_label = combo

            node_operation = best_combination[2]

            G.add_node(max_id, label=node_operation, type="operation")

            operator_id = max_id
            max_id += 1

            # Track which edge labels are already used for this operator
            used_labels = set()
            existing_edges = list(G.in_edges(operator_id, data=True))
            for edge in existing_edges:
                if 'label' in edge[2]:
                    used_labels.add(edge[2]['label'])

            for i in [best_combination[0], best_combination[1]]:
                if i < 0:
                    label = ">> " + str(abs(i))
                else:
                    label = "<< " + str(abs(i))

                if i == best_combination[0]:
                    if "left" not in used_labels:
                        edge_label = "left"
                    elif "right" not in used_labels:
                        edge_label = "right"
                    else:
                        edge_label = None
                else:
                    if "right" not in used_labels:
                        edge_label = "right"
                    elif "left" not in used_labels:
                        edge_label = "left"
                    else:
                        edge_label = None

                G.add_node(max_id, label=label, type="shift")
                G.add_edge(parent_node, max_id)
                if edge_label is not None:
                    G.add_edge(max_id, operator_id, label=edge_label)
                else:
                    G.add_edge(max_id, operator_id)
                
                if edge_label is not None:
                    used_labels.add(edge_label)
                max_id += 1

            if child_node is not None:
                if child_edge_label is not None:
                    G.add_edge(operator_id, child_node, label=child_edge_label)
                else:
                    G.add_edge(operator_id, child_node)

    remove_vals = list(set(remove_vals))
    remove_nodes = list(set(remove_nodes))

    for node in remove_vals:
        if node in G.nodes():
            G.remove_node(node)
    
    for item in remove_nodes:
        if item in G.nodes():
            G.remove_node(item)

    G = relabel_graph(G)
    G = remove_0_bit_shifts(G)

    return G




def closest_power_of_2(coeff):
    best_sum = None
    closest_diff = float('inf')
    best_combination = None
    
    max_exp = 20 
    min_exp = -10  

    # Iterate through all possible pairs of powers of 2
    for exp1 in range(min_exp, max_exp + 1):
        for exp2 in range(min_exp, max_exp + 1):
            power1 = 2 ** exp1
            power2 = 2 ** exp2

            
            # Sum 
            power_sum = power1 + power2
            diff_sum = abs(coeff - power_sum)
            if diff_sum < closest_diff:
                closest_diff = diff_sum
                best_sum = power_sum
                best_combination = (exp1, exp2, '+')

            # Difference 
            power_diff = power1 - power2
            diff_diff = abs(coeff - power_diff)
            if diff_diff < closest_diff:
                closest_diff = diff_diff
                best_sum = power_diff
                best_combination = (exp1, exp2, '-')

    return best_combination, best_sum


def remove_0_bit_shifts(G):
    remove_nodes = []
    new_edge = []
    for node in G.nodes():
        if G.nodes[node]['label'] == "<< 0":
            #print(G.nodes[node]['label'])

            in_edge = list(G.in_edges(node))[0][0]
            out_edge = list(G.out_edges(node))[0][1]
            out_edge_label = list(G.out_edges(node, data = True))[0][2]['label']

            #print(node)
            remove_nodes.append(node)
            new_edge.append((in_edge, out_edge, out_edge_label))

    for item in remove_nodes:
        G.remove_node(item)
    
    for j in new_edge:
        G.add_edge(j[0], j[1], label = j[2])

    return G


def make_color_assignments(dsp_solutions):
    color_assignments = {}
    color_list = cc.glasbey_light[:len(dsp_solutions)]
    for m in range(0, len(dsp_solutions)):
        for j in dsp_solutions[m]:
            color_assignments[j] = color_list[m]

    return color_assignments



def postadder_last_node_in_dsp(G, node, dsp_combos):
    for combo in dsp_combos:
        operations = [G.nodes[s]['label'] for s in combo]
        if '*' in operations:
            
            #print(operations)
            if node == combo[-1] and G.nodes[combo[-1]]['label'] != '*':
                # print("End of DSP with mult")
                return True
        
    return False


def add_postadder_shift_nodes(G, frac_bit_num, dsp_combos, adjusted_levels):

    if frac_bit_num != 0:
        add_node_between = []
        for combo in dsp_combos:
            for node in combo:
                if postadder_last_node_in_dsp(G, node, dsp_combos):
                    # print(combo, node)

                    edges_in = G.in_edges(node, data = True)

                    for edge in edges_in:
                        if edge[0] not in combo:
                            add_node_between.append(edge)
                    
        #print("here")
        # print(add_node_between)
        # print("Edges: ", add_node_between)
        node_num = max(G.nodes()) + 1
        for edge in add_node_between:
            descendants = set()
        
            # The shifted result is needed right after and so all nodes need to be moved one down
            if abs(adjusted_levels[edge[0]] - adjusted_levels[edge[1]]) == 1:
                # Find the DSP matches that are on the same level
                dsp_matched = [sublist for sublist in dsp_combos if any(adjusted_levels[x] == adjusted_levels[edge[1]] for x in sublist)]
                dsp_matched = [item for sublist in dsp_matched for item in sublist]
                # print(dsp_matched)
                # print("Shifted result is needed right after")

                smallest_level = np.inf
                for t in dsp_matched:
                    if adjusted_levels[t] < smallest_level:
                        smallest_level = adjusted_levels[t]
                # print(smallest_level)

                for item, level in adjusted_levels.items():
                    if level > smallest_level or (level == smallest_level and item in dsp_matched):
                        adjusted_levels[item] += 1
                        # print()


            G.remove_edge(edge[0], edge[1])
            label = "<< " + str(frac_bit_num)
           

            G.add_node(node_num, label= label, type="pshift")
            G.add_edge(edge[0], node_num)

            G.add_edge(node_num, edge[1], label = edge[2]['label'])

            node_num += 1
    
    return G, adjusted_levels



def only_bitshifts(G, new_levels, level):
    for j in level:
        nodes_in_that_level = [w for w, v in new_levels.items() if v == j]
        labels = [G.nodes[s]['label'] for s in nodes_in_that_level]

        if not all(('<<' in s or '>>' in s) for s in labels):
            return False
    return True


def move_dsp_components(node_levels, dsp_combo):
    dsp_combo_reversed = dsp_combo[::-1]


    dsp_combo_levels = []
    
    for w, v in enumerate(dsp_combo_reversed):
        dsp_combo_levels.append(node_levels[v])
    
    max_level = dsp_combo_levels[0]
    # print("Max Level: ", max_level)

    new_combo_levels = []

    new_combo_levels.append(max_level)
    for i in range(1, len(dsp_combo_levels)):
        new_combo_levels.append(max_level - i)

    new_combo_levels = new_combo_levels[::-1]


    for i in range(0, len(new_combo_levels)):
        node_levels[dsp_combo[i]] = new_combo_levels[i]


    return node_levels




def rearrange_dsps(G, initial_levels, dsp_combos):
    adjusted_levels = initial_levels
    for j in dsp_combos:
        if len(j) > 1:

            related_levels = sorted([adjusted_levels[i] for i in j])

            # The levels between nodes in a DSP block
            levels_to_check = []
            for k in range(min(related_levels) + 1, max(related_levels)):
                if k in related_levels:
                    continue
                # Creates list of levels that need to be checked
                levels_to_check.append(k)

            if len(levels_to_check) != 0:
                # print("The levels in between: ", levels_to_check)
                adjusted_levels = move_dsp_components(adjusted_levels, j)


    return adjusted_levels






def find_dsp_combos_in_levels(dsp_the_node_begins, split_stage_levels, dsp_combos):
    dsp_levels = set(split_stage_levels[node] for node in dsp_the_node_begins)
    
    combos_in_levels = []
    
    for combo in dsp_combos:
        # Skip DSP combo itself
        if combo == dsp_the_node_begins:
            continue
            
        # Check if any node in the combo is in the levels
        for node in combo:
            if split_stage_levels[node] in dsp_levels:
                if combo not in combos_in_levels:
                    combos_in_levels.append(combo)
                break 
    
    return combos_in_levels





def split_into_dsp_stages(G, adjusted_levels, dsp_combos):

    split_stage_levels = dict(adjusted_levels)
    max_iterations = len(G.nodes()) * 3
    iteration_count = 0
    
    dsp_dependencies = find_the_dependencies(G, dsp_combos, split_stage_levels)
    
    made_changes = True
    while made_changes and iteration_count < max_iterations:
        iteration_count += 1
        made_changes = False
        
        # DSP combos need to be well timed
        for i, dsp_combo in enumerate(dsp_combos):
            if enforce_dsp_timing_constraints(G, dsp_combo, split_stage_levels, dsp_combos, i):
                made_changes = True
        
        pipeline_analysis = analyse_pipeline_levels(G, split_stage_levels, dsp_combos)
        
        for level_num in sorted(pipeline_analysis.keys()):
            level_info = pipeline_analysis[level_num]
            
            if level_info['mixed_nodes']:
                conflicting_dsps = level_info['starting_dsps']
                
                for dsp_combo in conflicting_dsps:
                    new_start_level = find_next_clean_level(dsp_combo, split_stage_levels, dsp_combos, dsp_dependencies, level_num, G)
                    
                    if new_start_level > split_stage_levels[dsp_combo[0]]:
                        move_dsp_combo_with_dependents(G, dsp_combo, new_start_level, split_stage_levels, dsp_combos)
                        made_changes = True
                
                if made_changes:
                    break
        
    
    # validate_final_schedule(G, split_stage_levels, dsp_combos)
    
    return G, split_stage_levels


def enforce_dsp_timing_constraints(G, dsp_combo, levels, all_dsp_combos, combo_index):

    current_start_level = levels[dsp_combo[0]]
    
    all_input_nodes = get_all_dsp_input_nodes(dsp_combo, G)
    
    # Latest completion level of all inputs
    latest_input_completion = -1
    
    for input_node in all_input_nodes:
        if input_node not in levels:
            continue
            
        # Check if input is part of a DSP
        input_combo = find_dsp_combo_containing_node(input_node, all_dsp_combos)
        
        if input_combo:
           
            input_completion = max(levels[n] for n in input_combo if n in levels)
        else:
            input_completion = levels[input_node]
        
        latest_input_completion = max(latest_input_completion, input_completion)
    
    
    # DSP must start after all its inputs are complete
    required_start_level = latest_input_completion + 1
    
    if current_start_level <= latest_input_completion:
    
        # Move DSP combo and all its descendants
        move_dsp_combo_with_dependents(G, dsp_combo, required_start_level, levels, all_dsp_combos)
        return True
    

    return False


def move_dsp_combo_with_dependents(G, dsp_combo, new_start_level, levels, all_dsp_combos):

    # Calculate shift
    old_start_level = levels[dsp_combo[0]]
    level_shift = new_start_level - old_start_level
    
    
    if level_shift <= 0:
        return
    
    dependent_nodes = find_all_dependent_nodes(G, dsp_combo, levels, all_dsp_combos)
    
    # All DSP combos that need to be moved
    affected_dsp_combos = set()
    for node in dependent_nodes:
        combo = find_dsp_combo_containing_node(node, all_dsp_combos)
        if combo:
            affected_dsp_combos.add(tuple(combo))
    
    
 
    for i, node in enumerate(dsp_combo):
        old_level = levels[node]
        new_level = new_start_level + i
        levels[node] = new_level
    
    # Move each DSP combo as a single thing
    for combo_tuple in affected_dsp_combos:
        combo = list(combo_tuple)

        # Find the current starting level
        current_start = min(levels[node] for node in combo)

        # Calculate new starting level
        new_combo_start = current_start + level_shift
        
        
        # Move the entire combo
        for i, node in enumerate(combo):
            old_level = levels[node]
            new_level = new_combo_start + i
            levels[node] = new_level
    

    all_combo_nodes = set()
    for combo in affected_dsp_combos:
        all_combo_nodes.update(combo)

    standalone_nodes = dependent_nodes - all_combo_nodes
    if standalone_nodes:
        for node in standalone_nodes:
            old_level = levels[node]
            new_level = old_level + level_shift
            levels[node] = new_level


def find_all_dependent_nodes(G, source_dsp_combo, levels, all_dsp_combos):

    dependent_nodes = set()
    
    
    # Gets all descendants 
    for node in source_dsp_combo:
        descendants = nx.descendants(G, node)
        dependent_nodes.update(descendants)
    
    # Removes nodes that are part of the DSP combo itself
    dependent_nodes -= set(source_dsp_combo)
    
    # include the entire DSP combo to maintain internal ordering
    dependent_dsp_combos = set()
    for node in list(dependent_nodes): 
        combo = find_dsp_combo_containing_node(node, all_dsp_combos)
        if combo:
            dependent_dsp_combos.add(tuple(combo))
    
    # Add all nodes
    for combo_tuple in dependent_dsp_combos:
        dependent_nodes.update(combo_tuple)
    
    return dependent_nodes


def get_all_dsp_input_nodes(dsp_combo, G):

    input_nodes = set()
    
    
    for dsp_node in dsp_combo:
        if dsp_node not in G.nodes():
            continue
        
        
        for pred in G.predecessors(dsp_node):
            if pred not in G.nodes():
                continue
                
            # Skip if predecessor is in the same DSP combo
            if pred in dsp_combo:
                continue
                
            # Ignore shift nodes, skip over them
            if G.nodes[pred]['type'] == 'shift':

                for shift_input in G.predecessors(pred):
                    if shift_input not in dsp_combo:
                        input_nodes.add(shift_input)
            else:
                # Direct input node
                input_nodes.add(pred)
    
    return input_nodes



def calculate_earliest_possible_level(dsp_combo, levels, dependencies, G):

    earliest = 0
        
    # All input nodes to the combo
    all_input_nodes = get_all_dsp_input_nodes(dsp_combo, G)
    
    all_dsp_combos = []
    for dep in dependencies.keys():
        all_dsp_combos.append(list(dep))
    all_dsp_combos.append(dsp_combo)
    
    for input_node in all_input_nodes:
        if input_node not in levels:
            continue
            
        input_dsp_combo = find_dsp_combo_containing_node(input_node, all_dsp_combos)
        
        if input_dsp_combo:

            # Entire DSP combo must be complete
            input_combo_levels = []
            for n in input_dsp_combo:
                if n in levels:
                    input_combo_levels.append(levels[n])
            
            if input_combo_levels:
                input_completion_level = max(input_combo_levels)
                earliest = max(earliest, input_completion_level + 1)
        else:

            earliest = max(earliest, levels[input_node] + 1)
    
    return earliest


def find_next_clean_level(dsp_combo, levels, all_dsp_combos, dependencies, min_level, G):
       
    # Calculate smallest possible level 
    earliest_level = calculate_earliest_possible_level(dsp_combo, levels, dependencies, G)
    start_search = max(min_level + 1, earliest_level)
    
    max_search_level = max(levels.values()) + len(all_dsp_combos) + 5
    
    for candidate_level in range(start_search, max_search_level + 1):
        if is_level_available_for_dsp(candidate_level, dsp_combo, levels, all_dsp_combos, G):
            return candidate_level
    
    new_level = max(levels.values()) + 1
    return new_level


def is_level_available_for_dsp(level, dsp_combo, levels, all_dsp_combos, G):
    
    # All first nodes of DSP combos
    first_dsp_nodes = {combo[0] for combo in all_dsp_combos if combo}
    
    all_input_nodes = get_all_dsp_input_nodes(dsp_combo, G)
    
    # Check that all input nodes are before
    for input_node in all_input_nodes:
        if input_node in levels:
            input_level = levels[input_node]
            if input_level >= level:
                return False
    
    # Check that starting level only has nodes that start DSP combos
    nodes_at_start_level = [node for node, node_level in levels.items() if node_level == level]
    for node_at_level in nodes_at_start_level:
        if node_at_level not in dsp_combo and node_at_level not in first_dsp_nodes:
            return False
    
    # DSP combo there should not create issues with other
    dsp_levels_needed = list(range(level, level + len(dsp_combo)))
    
    for other_combo in all_dsp_combos:
        if tuple(other_combo) == tuple(dsp_combo):
            continue
            
        for i, other_node in enumerate(other_combo):
            if other_node not in levels:
                continue
                
            other_node_level = levels[other_node]
            other_combo_start = other_node_level - i
            other_levels_needed = range(other_combo_start, other_combo_start + len(other_combo))
            
            # Checks for overlap
            for our_level in dsp_levels_needed:
                if our_level in other_levels_needed:
                    our_position = our_level - level
                    other_position = our_level - other_combo_start
                    
                    if not (our_position == 0 and other_position == 0):
                        return False
    
    return True


def find_the_dependencies(G, dsp_combos, levels):

    dependencies = {}
    
    for dsp_combo in dsp_combos:
        dependencies[tuple(dsp_combo)] = set()
        
        # All inputs to the DSP combo
        for node in dsp_combo:
            for pred in G.predecessors(node):
                if pred in G.nodes() and G.nodes[pred]['type'] != 'shift':

                    # DSP combo the predecessor is part of
                    pred_dsp = find_dsp_combo_containing_node(pred, dsp_combos)
                    if pred_dsp and tuple(pred_dsp) != tuple(dsp_combo):
                        dependencies[tuple(dsp_combo)].add(tuple(pred_dsp))
    
    return dependencies


def analyse_pipeline_levels(G, levels, dsp_combos):

    pipeline_analysis = {}
    
    nodes_by_level = {}
    for node, level in levels.items():
        if level not in nodes_by_level:
            nodes_by_level[level] = []
        nodes_by_level[level].append(node)
    
    # analyse each level
    for level, nodes in nodes_by_level.items():
        operation_nodes = [n for n in nodes if n in G.nodes() and G.nodes[n]['type'] == 'operation' and '<<' not in G.nodes[n]['label'] and '>>' not in G.nodes[n]['label']]
        
        # which dsp combo starts in this level
        starting_dsps = []
        continuing_dsps = []
        first_dsp_nodes = {combo[0] for combo in dsp_combos if combo}
        
        for node in operation_nodes:
            if node in first_dsp_nodes:
                dsp_combo = find_dsp_combo_containing_node(node, dsp_combos)
                if dsp_combo:
                    starting_dsps.append(dsp_combo)
            else:
                dsp_combo = find_dsp_combo_containing_node(node, dsp_combos)
                if dsp_combo:
                    continuing_dsps.append(dsp_combo)
        
        pipeline_analysis[level] = {
            'operation_nodes': operation_nodes,
            'starting_dsps': starting_dsps,
            'continuing_dsps': continuing_dsps,
            'mixed_nodes': (len(starting_dsps) > 0) and (len(continuing_dsps) > 0),
        }
    
    return pipeline_analysis


def find_dsp_combo_containing_node(node, dsp_combos):

    for combo in dsp_combos:
        if node in combo:
            return combo
    return None



def isolate_shift_nodes(G, split_shift_levels, dsp_combos):
    divide_shift_levels = dict(split_shift_levels)
    shift_nodes = []


    for node in G.nodes():
        if ('<<' in G.nodes[node]['label'] or '>>' in G.nodes[node]['label']):
            shift_nodes.append(node)

    shift_nodes = sorted(shift_nodes, key=lambda x: divide_shift_levels[x])

    # For each shift node, need to find its child -> DSP combo child is in -> level of first node in that DSP combo

    for node in shift_nodes:
        child = list(G.out_edges(node))
        child_node = child[0][1]
        # print("Child node: ", child_node)

        # equal_level_nodes =  [k for k, v in divide_shift_levels.items() if v == divide_shift_levels[child_node]]
        associated_combos = [sub for sub in dsp_combos if child_node in sub][0]
        first_node_in_stage = associated_combos[0]


        equal_level_nodes =  [k for k, v in divide_shift_levels.items() if v == divide_shift_levels[node]]
        only_shift_nodes = all('<<' in G.nodes[node]['label'] or '>>' in G.nodes[node]['label'] for node in equal_level_nodes)
        
        #if not only_shift_nodes:
        divide_shift_levels[node] = divide_shift_levels[first_node_in_stage]

        # Node should go on the same level as the first node in the associated combo


    ## Puts the shift nodes on their own line
    ## After having moved shift nodes to the beginning, go through and move nodes on the same line as the shift down

    current_level = min(divide_shift_levels.values())
    max_level = max(divide_shift_levels.values())
    
    while current_level <= max_level:
        equal_level_nodes =  [k for k, v in divide_shift_levels.items() if v == current_level]

        contains_shift_nodes = any('<<' in G.nodes[node]['label'] or '>>' in G.nodes[node]['label'] for node in equal_level_nodes)
        only_shift_nodes = all('<<' in G.nodes[node]['label'] or '>>' in G.nodes[node]['label'] for node in equal_level_nodes)
        
        operation_nodes = []
        descendants = set()
       

        if contains_shift_nodes and not only_shift_nodes:
            for node in G.nodes():
                if not ('<<' in G.nodes[node]['label'] or '>>' in G.nodes[node]['label']) and divide_shift_levels[node] == current_level:
                    operation_nodes.append(node)

            nodes_below = [k for k, v in divide_shift_levels.items() if v > current_level]


            descendants.update(set(operation_nodes))
            descendants.update(set(nodes_below))

            for item in descendants:
                divide_shift_levels[item] = divide_shift_levels[item] + 1

            max_level = max(divide_shift_levels.values())


        if only_shift_nodes:
            # Remove gaps in stage numbers
            for node in divide_shift_levels:
                if divide_shift_levels[node] > current_level:
                    divide_shift_levels[node] -= 1
            max_level -= 1  
        else:
            current_level += 1 

    return divide_shift_levels


def find_dsp_start_levels(dsp_combos, node_levels, level_a, level_b):
    dsp_start_levels = set()
    
    for combo in dsp_combos:
        if len(combo) > 0:
            # Find first node in the DSP combination
            first_node = min(combo, key=lambda node: node_levels[node])
            first_node_level = node_levels[first_node]
            
            if level_a < first_node_level < level_b:
                dsp_start_levels.add(first_node_level)
    
    return sorted(dsp_start_levels)




def add_registers(G, node_levels, dsp_combos, unroll):

    register_edges = []
    
    for edge in G.edges():
        source, target = edge
        
        if (node_levels[target] - node_levels[source] <= 1 or 
            any(source in combo and target in combo for combo in dsp_combos)):
            continue
            
        target_combo = next((combo for combo in dsp_combos if target in combo), None)
        if target_combo:
            target_start_level = node_levels[target_combo[0]]
        else:
            target_start_level = node_levels[target]
        
        # Required register levels
        needed_levels = find_dsp_start_levels(dsp_combos, node_levels, node_levels[source], target_start_level)
        
        if needed_levels:
            edge_data = G.get_edge_data(source, target)
            edge_label = edge_data['label'] if edge_data else None
            
            register_edges.append({
                'source': source,
                'target': target,
                'edge_label': edge_label,
                'needed_levels': needed_levels,
                'target_start_level': target_start_level
            })
    
    for edge_info in register_edges:
        G.remove_edge(edge_info['source'], edge_info['target'])
    
    # Group register requirements by source node
    source_groups = {}
    for edge_info in register_edges:
        source = edge_info['source']
        if source not in source_groups:
            source_groups[source] = []
        source_groups[source].append(edge_info)
    
    # Maximum is the max plus 1
    max_node = max(G.nodes()) + 1
    
    for source_node, edge_infos in source_groups.items():

        all_levels = set()
        for edge_info in edge_infos:
            all_levels.update(edge_info['needed_levels'])
        sorted_levels = sorted(all_levels)
        
        # Adds one register per level
        register_chain = {}
        prev_node = source_node
        
        for level in sorted_levels:
            reg_node = max_node
            G.add_node(reg_node, label="1", type="register")
            G.add_edge(prev_node, reg_node)
            node_levels[reg_node] = level
            register_chain[level] = reg_node
            prev_node = reg_node
            max_node += 1
        
        # Create the chain of registers
        for edge_info in edge_infos:
            target = edge_info['target']
            edge_label = edge_info['edge_label']
            # Highest level where the data needs to go to
            # This is important for it to keep registers together
            max_needed_level = max(edge_info['needed_levels'])
            
            # Connect registers to the node
            # If a register needs to go through the same level it should attach to one that already exists
            if max_needed_level in register_chain:
                # Connect it to register that is already there
                reg_node = register_chain[max_needed_level]
                if edge_label:
                    G.add_edge(reg_node, target, label=edge_label)
                else:
                    G.add_edge(reg_node, target)
    
    if not unroll:
        merge_consecutive_registers(G, node_levels)
    
    return G, node_levels


def merge_consecutive_registers(G, node_levels):
    
    register_nodes = [node for node in G.nodes() if G.nodes[node]['type'] == 'register']
    processed = set()
    
    for reg_node in register_nodes:
        if reg_node in processed or reg_node not in G.nodes():
            continue
            
        chain = [reg_node]
        current = reg_node
        
        while True:
            successors = list(G.successors(current))
            
            # Stop if not one successor or successor isnt a register
            if (len(successors) != 1 or 
                successors[0] not in G.nodes() or 
                G.nodes[successors[0]]['type'] != 'register'):
                break
                
            next_reg = successors[0]
            chain.append(next_reg)
            current = next_reg
        
        # Merge chain if has multiple registers
        if len(chain) > 1:
            first_reg = chain[0]
            last_reg = chain[-1]
            
            # total register count
            total_count = len(chain)
            
            G.nodes[first_reg]['label'] = f"{total_count}"
            node_levels[first_reg] = node_levels[last_reg]
            
            final_edges_data = []
            for target in G.successors(last_reg):
                edge_data = G.get_edge_data(last_reg, target)
                final_edges_data.append((target, edge_data))

            # Remove inter registers
            for reg in chain[1:]:
                G.remove_node(reg)
                node_levels.pop(reg, None)
                processed.add(reg)

            # Connect first register to final targets using collected data
            for target, edge_data in final_edges_data:
                if edge_data and 'label' in edge_data:
                    G.add_edge(first_reg, target, label=edge_data['label'])
                else:
                    G.add_edge(first_reg, target)
        
        processed.add(reg_node)



def count_dsp_stages(dsp_combos, node_levels):

    # Get the starting level of each DSP combo (level of first node in combo)
    dsp_start_levels = set()
    
    for combo in dsp_combos:
        # Find the first node 
        first_node = min(combo, key=lambda node: node_levels[node])
        start_level = node_levels[first_node]
        dsp_start_levels.add(start_level)
    
    return len(dsp_start_levels)

def DSPBlockNumber(user_input):

    poly = PolynomialParser(user_input)
    G = nx.DiGraph()
    node_id, root_id = process_expression(poly.expr_tree, G, 0)

    
    G_mod = implement_bit_shifts(G)
  
    initial_levels = GenerateTree(G_mod, user_input, display = False).stage_levels(G_mod)

    DSPSearch = DSPSolver(G_mod, initial_levels)
    dsp_combos = DSPSearch.Solver

    return len(dsp_combos)



def DSPBlockNumberWithoutOptimization(user_input):

    poly = PolynomialParser(user_input)
    G = nx.DiGraph()
    node_id, root_id = process_expression(poly.expr_tree, G, 0)
  
    initial_levels = GenerateTree(G, user_input, display = True).stage_levels(G)

    DSPSearch = DSPSolver(G, initial_levels)
    dsp_combos = DSPSearch.Solver

    

    return G, dsp_combos


def GenerateGraph(user_input, frac_bit_num, show_graph):

    poly = PolynomialParser(user_input)
    G = nx.DiGraph()
    node_id, root_id = process_expression(poly.expr_tree, G, 0)

    G_mod = implement_bit_shifts(G)

    initial_levels = GenerateTree(G_mod, user_input, display = False).stage_levels(G_mod)


    DSPSearch = DSPSolver(G_mod, initial_levels)
    dsp_combos = DSPSearch.Solver

    initial_levels = rearrange_dsps(G_mod, initial_levels, dsp_combos)

    G_mod, split_stage_levels = split_into_dsp_stages(G_mod, initial_levels, dsp_combos)
  
    split_stage_levels = isolate_shift_nodes(G_mod, split_stage_levels, dsp_combos)

    G_mod, split_stage_levels = add_registers(G_mod, split_stage_levels, dsp_combos, unroll = False)
    
    GenerateTree(G_mod, user_input, specific_node_colors= make_color_assignments(dsp_combos), provided_levels = split_stage_levels, display = show_graph)

    return G_mod, dsp_combos, split_stage_levels




def GeneratePrecisionGraph(user_input, frac_bit_num, show_graph, quantised, b_port_shifts):

    poly = PolynomialParser(user_input)
    G = nx.DiGraph()
    node_id, root_id = process_expression(poly.expr_tree, G, 0)

    G_mod = implement_bit_shifts(G)

    initial_levels = GenerateTree(G_mod, user_input, display = False).stage_levels(G_mod)


    DSPSearch = DSPSolver(G_mod, initial_levels)
    dsp_combos = DSPSearch.Solver


    split_stage_levels = rearrange_dsps(G_mod, initial_levels, dsp_combos)

    G_mod, split_stage_levels = split_into_dsp_stages(G_mod, split_stage_levels, dsp_combos)
  
    split_stage_levels = isolate_shift_nodes(G_mod, split_stage_levels, dsp_combos)

    # GenerateTree(G_mod, user_input, specific_node_colors= make_color_assignments(dsp_combos), provided_levels = split_stage_levels, display = True)


    if quantised:
        # print("It is quantised")

        if frac_bit_num > 0:
            frac_bits = frac_bit_num - b_port_shifts
            G_mod, split_stage_levels = add_postadder_shift_nodes(G_mod, frac_bits, dsp_combos, initial_levels)



    GenerateTree(G_mod, user_input, specific_node_colors= make_color_assignments(dsp_combos), provided_levels = split_stage_levels, display = show_graph)

    return G_mod, dsp_combos, split_stage_levels



def CreateTree(user_input, dsp_combos, G_mod, split_shift_levels):
    GenerateTree(G_mod, user_input, specific_node_colors= make_color_assignments(dsp_combos), provided_levels = split_shift_levels, display = True)



