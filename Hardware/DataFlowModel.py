#!/usr/bin/env python
# coding: utf-8

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


##----- FILE IMPORTS ---------------------

from PolynomialClass import *

## -----------------------------------

# Process the expression recursively
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
                    
                    left_id = parent_node  # for chaining operations
                
                return node_id, left_id


    # Power operation
    elif type(expr) == Power:
        base, exp = expr.args
        if hasattr(exp, 'is_integer') and exp.is_integer and exp > 1:
            return convert_power_to_multiplication(expr, base, exp, graph, node_id, parent_id, expr_to_id)

            
    #  variables
    elif type(expr) == Variable or hasattr(expr, 'is_symbol') and expr.is_symbol():
        curr_id = node_id
        graph.add_node(curr_id, label=str(expr), type="value")
        expr_to_id[expr_hash] = curr_id
        node_id += 1
        if parent_id is not None:
            graph.add_edge(curr_id, parent_id)
        return node_id, curr_id
    
    #  constants
    elif hasattr(expr, 'is_number') and expr.is_number():
        curr_id = node_id
        # Display negative numbers directly with the negative sign
        graph.add_node(curr_id, label=str(expr), type="value")
        expr_to_id[expr_hash] = curr_id
        node_id += 1
        if parent_id is not None:
            graph.add_edge(curr_id, parent_id)
        return node_id, curr_id
    
    #  unary minus
    elif type(expr) == UnaryMinus:

        mult_node = node_id
        graph.add_node(mult_node, label="*", type="operation")
        expr_to_id[expr_hash] = mult_node
        node_id += 1
        
        if parent_id is not None:
            graph.add_edge(mult_node, parent_id)
        
        # Create -1 constant node
        const_node = node_id
        graph.add_node(const_node, label="-1", type="value")
        node_id += 1
        graph.add_edge(const_node, mult_node, label='left')
        
        # Process the term inside the unary minus
        node_id, term_id = process_expression(expr.value, graph, node_id, mult_node, top_level=False, expr_to_id=expr_to_id)
        graph.add_edge(term_id, mult_node, label='right')
        
        return node_id, mult_node
    
    #  subtraction
    elif type(expr) == Subtraction:
        sub_node = node_id
        graph.add_node(sub_node, label="-", type="operation")
        expr_to_id[expr_hash] = sub_node
        node_id += 1
        
        if parent_id is not None:
            graph.add_edge(sub_node, parent_id)
        
        # Process left and right terms
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

    # Handles base cases
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

    # For larger exponents uses exponentiation by squaring
    if exp % 2 == 0:
        left_exp = right_exp = exp // 2
    else:
        left_exp = 1
        right_exp = exp - 1

    # Builds left subtree
    left_expr = base ** left_exp
    if hash(left_expr) in expr_to_id:
        left_id = expr_to_id[hash(left_expr)]
    else:
        node_id, left_id = convert_power_to_multiplication(left_expr, base, left_exp, graph, node_id, None, expr_to_id)

    # Builds right subtree
    right_expr = base ** right_exp
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


# ------------------------------------------------------------------


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
            plt.figure(figsize=(15, 12))
            
            #Divide up the nodes
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
                text.set_zorder(3)  # labels on top
                   
            labels = {}
            for node in graph.nodes():
                attr_value = graph.nodes[node]['label']
                labels[node] = f"{node}: {attr_value}"
                #labels[node] = f"{attr_value}"
            
            graph_labels = nx.draw_networkx_labels(graph, pos, labels=labels, font_size=14, font_weight='bold')
            for text in graph_labels.values():
                text.set_zorder(3)  # labels on top

            plt.title(f"Node Graph for: ${(polynomial_str.replace('**', '^')).replace('*', '')}$", fontsize=16)

            # plt.title(f"Node Graph for", fontsize=16)
            plt.axis('off')
            plt.tight_layout()
            plt.show() 


    # Updates the levels with any new added nodes
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
                    # print(j)
                    # print(expanded_levels[node])
                    expanded_levels[j] = expanded_levels[node] - 1

        # print(expanded_levels)

        return expanded_levels


    # Orders nodes based on stages starting from the symbolic characters
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

        # print("Levels are: ", levels)
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
        
        #levels = self.stage_levels(G)
        nodes_by_level = self.organise_nodes_by_level(levels)
        
        max_level = max(nodes_by_level)
        spacing = 1
        
        for level, nodes in nodes_by_level.items():
            num_nodes = len(nodes)
            #print("Number of nodes: ", num_nodes)
            
            if level == 0:
                max_span = (num_nodes - 1) * spacing   
                #print("Max span: ", max_span) 
                x_start = -max_span / 2
                for i, node in enumerate(sorted(nodes)):
                    x = x_start + i * spacing
                    y = max_level - level
                    pos[node] = (x, y)
                    
            else:
                if(num_nodes == 1):
                    x_start =  random.uniform(-0.2 * max_span, 0.2 * max_span)
                else:
                    span = (num_nodes - 1) * (max_span/num_nodes)
                    x_start = - span / 2
                    
                for i, node in enumerate(sorted(nodes)):
                    x = x_start + i * (max_span/num_nodes) + random.uniform(-0.02 * max_span, 0.02 * max_span)
                    y = max_level - level
                    pos[node] = (x, y)     
        
        return pos
    
    

class DSPSolver:
    def __init__(self, graph, node_levels):
        self.original_graph = graph
        self.node_levels = node_levels
        self.dsp_combos = self.generate_valid_combinations()
        self.graph, self.BitShift_graph = self.remove_non_dsp_nodes(graph)
        

    # Generates valid combinations for the DSP48E1
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
            #print(label, check_numeric(label), node_type)

        for node in remove_nodes:
            G_new.remove_node(node)

        #GenerateTree(G_new, "Without bit shifts")

        return G_new, G

    # Finds the possible combinations in the graph
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
                        added = False

                        for m in range(len(operations)):
                            if operations[m] == '-' and m > 0:
                                added = True
                                edge = list(G.in_edges(path[m], data=True))
                                # print(edge)
                                if path[m-1] == edge[0][0] and path[m] == edge[0][1] and edge[0][2]['label'] == 'right':
                                    possible_combos.append(path)

                        if not added:
                            possible_combos.append(path)

                if len(path) > 3:
                    continue

                neighbours = list(G.successors(current))
                for k in neighbours:
                    if k not in path:
                        queue.append((k, path + [k]))

        # Checks for duplicates
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
                        # print("Combo being skipped: ", combo)
                        continue

                    if node in list(G_with_bitshifts.successors(grandparent)):
                        valid_combo = False
                        break
                
                if not valid_combo:
                    # print("Not valid combo: ", combo)
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
                    if G.nodes[combo[1]]['label'] == '*' and len(edges_in[0]) < 2:
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

    #  Solving is defined as a property of the class
    # Implementation of problem as ILP and greedy algorithm as fallback
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

        # Define constraint
        for node in self.graph.nodes():

            combos_with_node = []
            for i, combo in enumerate(specific_combos):
                if node in combo:
                    combos_with_node.append(i)

            problem += pu.lpSum(binary_dsp_combos[k] for k in combos_with_node) == 1

        solver = pu.PULP_CBC_CMD(timeLimit=5, msg = False)
        # Stops after finding one solution
        result = problem.solve(solver)

        if pu.LpStatus[result] != 'Optimal':
            print("No optimal")
            return self.GreedySolver()
        
        
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

    # Apply all changes
    for i in nodes_to_remove:
        # Remove all edges connected to the node
        G_mod.remove_edges_from(list(G.in_edges(i)))  # Remove incoming edges
        G_mod.remove_edges_from(list(G.out_edges(i)))  # Remove outgoing edges
        
        # Remove the node itself
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

    # Relabel nodes
    G_relabelled = nx.relabel_nodes(G, new_labels, copy=True)

    return G_relabelled


def implement_bit_shifts(G):
    # Remove the node with the value
    remove_vals = [] 
    # Contains the multiplication node. Remove the nodes and its edges
    remove_nodes = []  
    # Each sub list has the structure: [(... Bit shifts ... , operation), parent node, child node, child edge label]
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
                # else:
                #     # Handle unlabeled edges - assign based on order
                #     if left_edge is None:
                #         left_edge = tup
                #     else:
                #         right_edge = tup
            
            if left_edge is None or right_edge is None:
                continue

            # If the left edge is not a number, then continue with the loop
            if not check_numeric(G.nodes[left_edge[0]]['label']):
                continue

            # Skip multiplication nodes with -1 (unary minus representations)
            if G.nodes[left_edge[0]]['label'] == '-1':
                continue
            
            val = G.nodes[left_edge[0]]['label']

            # Adds the node that is the number
            remove_vals.append(left_edge[0]) 

            # Adds multiplication node whose edges will be removed
            remove_nodes.append(node) 
            parent_node = right_edge[0]
            
            # Get child node and its edge label
            child_edges = list(G.out_edges(node, data=True))
            if child_edges:
                child_node = child_edges[0][1]
                child_edge_label = child_edges[0][2]['label']
            else:
                child_node = None
                child_edge_label = None

            if float(val) > 0:
                if math.log2(float(val)).is_integer():
                    shift_val = math.log2(float(val))
                    single_bit_shift_nodes.append([shift_val, parent_node, child_node, child_edge_label])
                else:
                    best_combination, best_sum = closest_power_of_2(float(val))
                    bit_shift_nodes.append([best_combination, parent_node, child_node, child_edge_label])

    # Runs if it is a sum combination of bit shift nodes
    if single_bit_shift_nodes:
        max_id = max(G.nodes()) + 1
        
        for k, series in enumerate(single_bit_shift_nodes):
            shift_val, parent_node, child_node, child_edge_label = series

            if shift_val >= 0:
                label = "<< " + str(shift_val)
            elif shift_val < 0:
                label = ">> " + str(-shift_val)

            G.add_node(max_id, label=label, type="shift")
            G.add_edge(parent_node, max_id)
            
            if child_node is not None:
                G.add_edge(max_id, child_node, label=child_edge_label)

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

                # Determine edge label while avoiding duplicates
                if i == best_combination[0]:
                    if "left" not in used_labels:
                        edge_label = "left"
                    elif "right" not in used_labels:
                        edge_label = "right"
                else:
                    if "right" not in used_labels:
                        edge_label = "right"
                    elif "left" not in used_labels:
                        edge_label = "left"

                G.add_node(max_id, label=label, type="shift")
                G.add_edge(parent_node, max_id)
                G.add_edge(max_id, operator_id, label=edge_label)
                
                # Add this label to used_labels for next iteration
                used_labels.add(edge_label)
                max_id += 1

            if child_node is not None:
                G.add_edge(operator_id, child_node, label=child_edge_label)

    ## Do not remove nodes that are of type value and have a child that is not a multiplication symbol

    for node in remove_vals:
        children = list(G.successors(node))
        children_operations = [G.nodes[s]['label'] for s in children]
        if '+' in children_operations or '-' in children_operations or G.nodes[node]['label'] == '-1':
            remove_vals.remove(node)

    G.remove_nodes_from(remove_vals)
    for item in remove_nodes:
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

            
            # Calculate sum (2^exp1 + 2^exp2)
            power_sum = power1 + power2
            diff_sum = abs(coeff - power_sum)
            if diff_sum < closest_diff:
                closest_diff = diff_sum
                best_sum = power_sum
                best_combination = (exp1, exp2, '+')

            # Calculate difference (|2^exp1 - 2^exp2|)
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

    if frac_bit_num > 0:
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
                dsp_matched = [sublist for sublist in dsp_combos if any(adjusted_levels.get(x) == adjusted_levels[edge[1]] for x in sublist)]
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

                # adjusted_levels[dsp_matched[0]] +=1

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
    # print("DSP combo that has other nodes in between: ", dsp_combo)
    # print("Current levels: ", node_levels)

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
    # print("DSP combo: ", dsp_combo)
    # print("New Levels: ", new_combo_levels)

    for i in range(0, len(new_combo_levels)):
        node_levels[dsp_combo[i]] = new_combo_levels[i]

    # print("New levels: ", node_levels)

    return node_levels


    


def rearrange_dsps(G, initial_levels, dsp_combos):
    adjusted_levels = initial_levels
    for j in dsp_combos:
        if len(j) > 1:
            # print("DSP combo: ", j)

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
                # print("DSP Blocks need to b aligned:", j)
                adjusted_levels = move_dsp_components(adjusted_levels, j)


    return adjusted_levels






def find_dsp_combos_in_levels(dsp_the_node_begins, split_stage_levels, dsp_combos):
    # Get all levels occupied by the target DSP
    dsp_levels = set(split_stage_levels[node] for node in dsp_the_node_begins)
    
    # Find DSP combos that have nodes in those levels
    combos_in_levels = []
    
    for combo in dsp_combos:
        # Skip the original DSP combo
        if combo == dsp_the_node_begins:
            continue
            
        # Check if any node in this combo is in the target levels
        for node in combo:
            if split_stage_levels[node] in dsp_levels:
                if combo not in combos_in_levels:
                    combos_in_levels.append(combo)
                break  # Found one node, no need to check rest of combo
    
    return combos_in_levels




# Splitting DSPs into stages
def split_into_dsp_stages(G, adjusted_levels, dsp_combos):
# Move DSP nodes in order to get DSPs to start at the same time
    split_stage_levels = dict(adjusted_levels)

    level_num = 1

    while level_num <= max(split_stage_levels.values()):
    	
        nodes_at_this_level = [key for key, value in split_stage_levels.items() if value == level_num]
        # print("Nodes at this level: ", nodes_at_this_level)

        #  Operation nodes at this level
        operation_set_this_level = {key for key in nodes_at_this_level if '<<' not in G.nodes[key]['label'] and '>>' not in G.nodes[key]['label']}

        non_first_dsp_nodes = []

        # Nodes that are the beginning of a DSP combo
        first_term_dsp_combos = {sublist[0] for sublist in dsp_combos if sublist}

        # Loop through the nodes that have operations
        for current_node in operation_set_this_level:
            # if ('<<' not in G.nodes[node]['label'] and '>>' not in G.nodes[node]['label']):
                
            # print("Node: ", current_node)
            # print("Depth of: ", level_num)
            # print("Level number: ", level_num)

            # Stores the DSP combo that the current node is at the beginning of
            dsp_the_node_begins = [sublist for sublist in dsp_combos if sublist and sublist[0] == current_node]
            


            if dsp_the_node_begins:
                dsp_the_node_begins = dsp_the_node_begins[0]
                # print("DSP combo with this node as first: ", dsp_the_node_begins)
                # print("First terms of DSPs are: ", first_term_dsp_combos)
                # print("Operations on this level: ", nodes_at_this_level, operation_set_this_level)

                #------------------

                # Need to check that all nodes coming into the DSP combo are in the levels above the first node
                
                # Other operation nodes on the same level
                other_nodes = list(operation_set_this_level - {current_node})
                # print("Other nodes on this level: ", other_nodes ) 

                dependant_nodes = set() # Nodes that are inputs to the DSP combination

                for k in dsp_the_node_begins:
                    # Add the parent nodes of the dsp combo
                    for m in list(G.in_edges(k)):
                        if G.nodes[m[0]]['type'] != 'shift':
                            if m[0] not in dsp_the_node_begins:
                                dependant_nodes.add(m[0])
                        else: ## Do this because shift node should be ignored and the parents of the shift should be looked at
                            parent_edges = list(G.in_edges(m[0]))
                            for b in parent_edges:
                                if b[0] not in dsp_the_node_begins:
                                    dependant_nodes.add(b[0])

                                                

                # print("Dependant nodes: ", dependant_nodes)

                # Gets all levels occupied by the DSP
                dsp_levels = [split_stage_levels[node] for node in dsp_the_node_begins]
                
                # Find all nodes in those dsp_levels
                nodes_in_dsp_levels = []
                for node, level in split_stage_levels.items():
                    if level in dsp_levels and node not in dsp_the_node_begins:
                        nodes_in_dsp_levels.append(node)

                # print("Nodes of other concurrent DSPs: ", nodes_in_dsp_levels)
                # Checks that nodes in the same levels as the DSP combo also depend on the DSP combo
                has_overlap = bool(set(dependant_nodes) & set(nodes_in_dsp_levels))
                # print("Has overlap: ", has_overlap)


                if has_overlap:
                    # Find the DSP combos that overlap in levels
                    overlapping_dsps = find_dsp_combos_in_levels(dsp_the_node_begins, split_stage_levels, dsp_combos)
                    biggest_level = 0

                    for item in overlapping_dsps:
                        # Finds the deepest node that is at the end of a DSP combo 
                        if item[-1] in dependant_nodes:
                            if split_stage_levels[item[-1]] > biggest_level:
                                biggest_level = split_stage_levels[item[-1]]

                    #         print(item, split_stage_levels[item[-1]])
                    # print("Last node of DSP block it overlaps: ", overlapping_dsps)
                    # print("Biggest level: ", biggest_level)
                    # print("Level of first node in DSP: ", split_stage_levels[current_node])

                    # Number of levels to move down in order to not have overlap
                    move_down = biggest_level - split_stage_levels[current_node] + 1
                    # print("Move down by : ", move_down)


                    # All nodes that depend on the current node
                    node_descendants = nx.descendants(G, current_node) | {current_node}
                    sorted_descendants = sorted(node_descendants, key=lambda x: split_stage_levels[x])

                    current_level = split_stage_levels[current_node]
                    consecutive_list = []
                    expected_level = current_level + 1
                    
                    # Creates a list of descendant nodes that are directly below the node
                    # Stops if there is more than one level in between
                    for descendant in sorted_descendants:
                        desc_level = split_stage_levels[descendant]
                        
                        if desc_level == expected_level:
                            consecutive_list.append(descendant)
                            expected_level += 1
                        elif desc_level - expected_level >= 1:
                            # Gap of 1 or more levels, stop here
                            break
                        else:
                            # Small gap skip to this level
                            consecutive_list.append(descendant)
                            expected_level = desc_level + 1


                    # All nodes of the DSP combos that the nodes are in also need to be added 
                    # So the descendant nodes need to be moved down, but so do the DSP combos the nodes belong to
                    for combo in dsp_combos:
                        # Check if any value in consecutive_list appears in this combo
                        if any(value in combo for value in consecutive_list):
                            consecutive_list.extend(combo)

                    # Remove duplicates
                    consecutive_list = list(set(consecutive_list))
                                        
                    # print("Consecutive list is: ", consecutive_list)

                    # Move the nodes down
                    for item in consecutive_list:
                        # print(item, split_stage_levels[item])
                        split_stage_levels[item] += move_down

                # print(current_node, first_term_dsp_combos)
                # -----------

        # print("Avoid overlap")
        # CreateTree(user_input, dsp_combos, G, split_stage_levels)

        # Having moved the nodes, the nodes present on the current level need to be recomputed
        new_nodes_at_this_level = [key for key, value in split_stage_levels.items() if value == level_num]
        new_operation_set_this_level = {key for key in new_nodes_at_this_level if '<<' not in G.nodes[key]['label'] and '>>' not in G.nodes[key]['label']}
        
        # Checks that all nodes in the level are the first nodes in a dsp combo
        all_present = new_operation_set_this_level.issubset(first_term_dsp_combos)
        # print("New operations on this level: ", new_nodes_at_this_level)
        # print("All nodes in the level are the beginning of a DSP block: ", all_present)

        if not all_present:

            # Nodes that are not the beginning of a DSP block on this level
            # These nodes need to be moved down
            non_first_dsp_nodes = list(new_operation_set_this_level & first_term_dsp_combos)
            # print("These nodes need to be moved down:", non_first_dsp_nodes)
            # print(list(new_operation_set_this_level & first_term_dsp_combos))
            # print(new_operation_set_this_level)
            # print(first_term_dsp_combos)


        lowest_level_node = np.inf
        for n in non_first_dsp_nodes:
            if split_stage_levels[n] < lowest_level_node:
                lowest_level_node = n
        # print("Lowest level node: ", lowest_level_node)


        if non_first_dsp_nodes:
            # Move all nodes 1 level down
            # print("Node and below needs to be moved down to beginning of DSP")
            for item, level in split_stage_levels.items():
                if level > split_stage_levels[lowest_level_node] and item not in non_first_dsp_nodes:
                    # print(item, current_node)
                    split_stage_levels[item] += 1

            for item in non_first_dsp_nodes:
                split_stage_levels[item] += 1

        # print("start at the same time")
        # CreateTree(user_input, dsp_combos, G, split_stage_levels)
                    
        level_num += 1

    return G, split_stage_levels





def isolate_shift_nodes(G, split_shift_levels, dsp_combos):
    divide_shift_levels = dict(split_shift_levels)
    shift_nodes = []


    for node in G.nodes():
        if ('<<' in G.nodes[node]['label'] or '>>' in G.nodes[node]['label']):
            shift_nodes.append(node)

    shift_nodes = sorted(shift_nodes, key=lambda x: divide_shift_levels[x])

    # For each shift node, need to find its child -> DSP combo child is in -> level of first node in that DSP combo

    # print(shift_nodes)
    for node in shift_nodes:
        # print("Shift Node: ", node)
        child = list(G.out_edges(node))
        child_node = child[0][1]
        # print("Child node: ", child_node)

        # equal_level_nodes =  [k for k, v in divide_shift_levels.items() if v == divide_shift_levels[child_node]]
        associated_combos = [sub for sub in dsp_combos if child_node in sub][0]
        first_node_in_stage = associated_combos[0]
        # print("Associated combo: ", node, associated_combos)


        equal_level_nodes =  [k for k, v in divide_shift_levels.items() if v == divide_shift_levels[node]]
        only_shift_nodes = all('<<' in G.nodes[node]['label'] or '>>' in G.nodes[node]['label'] for node in equal_level_nodes)
        # print(node, only_shift_nodes)
        
        #if not only_shift_nodes:
        divide_shift_levels[node] = divide_shift_levels[first_node_in_stage]

        # Node should go on the same level as the first node in the associated combo


    ## Puts the shift nodes on their own line
    ## After having moved shift nodes to the beginning, go through and move nodes on the same line as the shift down

    current_level = min(divide_shift_levels.values())
    max_level = max(divide_shift_levels.values())
    
    while current_level <= max_level:
        equal_level_nodes =  [k for k, v in divide_shift_levels.items() if v == current_level]
        # print("Nodes of equal level: ", equal_level_nodes)

        contains_shift_nodes = any('<<' in G.nodes[node]['label'] or '>>' in G.nodes[node]['label'] for node in equal_level_nodes)
        only_shift_nodes = all('<<' in G.nodes[node]['label'] or '>>' in G.nodes[node]['label'] for node in equal_level_nodes)
        
        # prev_level_nodes =  [k for k, v in divide_shift_levels.items() if v == current_level -1]
        # prev_only_shift_nodes = all('<<' in G.nodes[node]['label'] or '>>' in G.nodes[node]['label'] for node in prev_level_nodes)
        
        operation_nodes = []
        descendants = set()
        # print("Contains shift nodes: ", contains_shift_nodes)
        # print("Has only shift nodes: ", only_shift_nodes)
        if contains_shift_nodes and not only_shift_nodes:
            for node in G.nodes():
                if not ('<<' in G.nodes[node]['label'] or '>>' in G.nodes[node]['label']) and divide_shift_levels[node] == current_level:
                    operation_nodes.append(node)

            # print("Operations on that level: ", operation_nodes)
            nodes_below = [k for k, v in divide_shift_levels.items() if v > current_level]

            # print(nodes_below)
            descendants.update(set(operation_nodes))
            descendants.update(set(nodes_below))

            for item in descendants:
                divide_shift_levels[item] = divide_shift_levels[item] + 1

            max_level = max(divide_shift_levels.values())


        if only_shift_nodes:
            # Move all higher levels up by 1 to compress this level
            for node in divide_shift_levels:
                if divide_shift_levels[node] > current_level:
                    divide_shift_levels[node] -= 1
            max_level -= 1  
        else:
            current_level += 1 
        # current_level += 1

    return divide_shift_levels


def find_dsp_start_levels(dsp_combos, node_levels, level_a, level_b):
    dsp_start_levels = set()
    
    for combo in dsp_combos:
        if len(combo) > 0:
            # Find the first node in the DSP combination (lowest level)
            first_node = min(combo, key=lambda node: node_levels[node])
            first_node_level = node_levels[first_node]
            
            # Check if this starting level is within the range
            if level_a < first_node_level < level_b:
                dsp_start_levels.add(first_node_level)
    
    return sorted(dsp_start_levels)



def add_registers(G, node_levels, dsp_combos):

    registers_dict = {}
    # Accumulate all edges going from one node to another
    all_edges = []
    for node in G.nodes():
        all_edges.extend(list(G.out_edges(node)))

    # Find the level each node is on
    for edge in all_edges:
        # Only look at non consecutive nodes and edges that are not in the same dsp combo
        if node_levels[edge[1]] - node_levels[edge[0]] > 1 and not any(edge[0] in combo and edge[1] in combo for combo in dsp_combos):
            # print(edge[0], edge[1])
                       
            # Find the level of the first node in the node's dsp combo which the end edge belongs to
            in_combo = [combo for combo in dsp_combos if edge[1] in combo]
            # print("In combo; ", in_combo)
            if in_combo:
                last_level = in_combo[0][0]
            else:
                last_level = edge[1]
            # print("The needed level: ", last_level)

            # print(with_register_levels[edge[0]], with_register_levels[edge[1]])

            needed_registers = find_dsp_start_levels(dsp_combos, node_levels, node_levels[edge[0]], node_levels[last_level])
            registers_dict[(edge[0], edge[1])] = needed_registers

    max_node = max(G.nodes()) + 1
    # print("MAX NODE IS: ", max_node)
    for item, values in registers_dict.items():
        if values:
            edge_label = G.get_edge_data(item[0], item[1])
            if edge_label:
                edge_label = edge_label['label']
            else:
                edge_label = None

            # print(edge_label)
            G.remove_edge(item[0], item[1])

            prev_node = item[0]
            for val in values:
                
                G.add_node(max_node, label = "R", type = "register")
                G.add_edge(prev_node, max_node)

                node_levels[max_node] = val
                prev_node = max_node
                max_node += 1
                
                # print(item)
            # print("Edge to add: ", prev_node, item[1])
            if edge_label is not None:
                G.add_edge(prev_node, item[1], label = edge_label)
            else:
                G.add_edge(prev_node, item[1])

    # print(registers_dict)
    # print(all_edges)
    return G, node_levels






def DSPBlockNumber(user_input):

    poly = PolynomialParser(user_input)
    G = nx.DiGraph()
    node_id, root_id = process_expression(poly.expr_tree, G, 0)

    
    G_mod = implement_bit_shifts(G)
  
    initial_levels = GenerateTree(G_mod, user_input, display = False).stage_levels(G_mod)


    DSPSearch = DSPSolver(G_mod, initial_levels)
    dsp_combos = DSPSearch.Solver
    # print("Number of DSP Blocks: ", len(dsp_combos))

    return len(dsp_combos)




def GenerateGraph(user_input, frac_bit_num, show_graph):

    poly = PolynomialParser(user_input)
    G = nx.DiGraph()
    node_id, root_id = process_expression(poly.expr_tree, G, 0)

    #G_mod = merge_negatives(G)
    G_mod = implement_bit_shifts(G)

    initial_levels = GenerateTree(G_mod, user_input, display = False).stage_levels(G_mod)


    DSPSearch = DSPSolver(G_mod, initial_levels)
    dsp_combos = DSPSearch.Solver
    # print("Number of DSP Blocks: ", len(dsp_combos))

    # # Displays tree and assigns pos to the variable

    initial_levels = rearrange_dsps(G_mod, initial_levels, dsp_combos)

    # G_mod, adjusted_levels = add_postadder_shift_nodes(G_mod, frac_bit_num, dsp_combos, initial_levels)

    G_mod, split_stage_levels = split_into_dsp_stages(G_mod, initial_levels, dsp_combos)
  
    split_stage_levels = isolate_shift_nodes(G_mod, split_stage_levels, dsp_combos)

    # G_mod, split_stage_levels = add_registers(G_mod, split_stage_levels, dsp_combos)

    GenerateTree(G_mod, user_input, specific_node_colors= make_color_assignments(dsp_combos), provided_levels = split_stage_levels, display = show_graph)

    return G_mod, dsp_combos, split_stage_levels




def GeneratePrecisionGraph(user_input, frac_bit_num, show_graph, quantised):

    poly = PolynomialParser(user_input)
    G = nx.DiGraph()
    node_id, root_id = process_expression(poly.expr_tree, G, 0)

    GenerateTree(G, user_input, display = True)

    #G_mod = merge_negatives(G)
    G_mod = implement_bit_shifts(G)

    initial_levels = GenerateTree(G_mod, user_input, display = False).stage_levels(G_mod)


    DSPSearch = DSPSolver(G_mod, initial_levels)
    dsp_combos = DSPSearch.Solver
    # print("Number of DSP Blocks: ", len(dsp_combos))

    # # Displays tree and assigns pos to the variable

    split_stage_levels = rearrange_dsps(G_mod, initial_levels, dsp_combos)

    G_mod, split_stage_levels = split_into_dsp_stages(G_mod, split_stage_levels, dsp_combos)
  
    split_stage_levels = isolate_shift_nodes(G_mod, split_stage_levels, dsp_combos)

    GenerateTree(G_mod, user_input, specific_node_colors= make_color_assignments(dsp_combos), provided_levels = split_stage_levels, display = show_graph)


    if quantised:
        print("IT is quantised")
        G_mod, split_stage_levels = add_postadder_shift_nodes(G_mod, frac_bit_num, dsp_combos, initial_levels)


    # G_mod, split_stage_levels = add_registers(G_mod, split_stage_levels, dsp_combos)

    GenerateTree(G_mod, user_input, specific_node_colors= make_color_assignments(dsp_combos), provided_levels = split_stage_levels, display = show_graph)

    return G_mod, dsp_combos, split_stage_levels





def CreateTree(user_input, dsp_combos, G_mod, split_shift_levels):
    GenerateTree(G_mod, user_input, specific_node_colors= make_color_assignments(dsp_combos), provided_levels = split_shift_levels, display = True)




# user_input = "((x*y*z)^2) -1.2*((x+z)^2)*y*(y- 1.3*x) - 7.8*(x - 6.1*z) + 5*((x + y) - x)"

# # user_input = "x^2 + (z - x*y)"

# #user_input = "-2.5*(x+z)^2 + (z - x*y) + (z*x - y) "

# # user_input = "-4.1 * x^2 + 5.3 * ( y - (x+y)*z) - 1.01 *y * (x+z) + 1.3*x - 3*(( z - (x+z)^2) - x*y)"

# # user_input = "1.5*x^2 + 1.6*(3.2*z - x*y )"

# user_input = "(((6.4*(4.5*x + (z^2) + 3*z)^2)^2) + (2.1*(x*w+z)) - (x*y*z^2) - 1.2*(x+z)^2*y*(y- 1.3*x) - 7.8*(x - 6.1*z) + 5*((x + y) - x)" ### Use to fix shift levels

# # user_input = "-4.5*(y - (x+y)*z) + 3.4*(x*y) + 9.8*(y - x^2+z)"

# # SINDy test
# # user_input = "(0.085) - (2.700 * x) + (2.911 * y) - (0.036 * z) - (0.469 * x^2) + (0.428 * y^2) - (0.448 * z^2) - (0.077 * x*y) - (0.635 * x*z) + (1.225 * y*z) - (1.475 * x + y*z) + (2.276 * y + x*z) - (0.113 * z + x*y) + (0.590 * (x + y)*z) + (1.148 * (x + z)*y) - (0.712 * (y + z)*x) - (2.110 * (x + y)*z + x) + (1.112 * (x + z)*y + z) - (0.748 * (y + z)*x + z) - (0.270 * ((y + z)^2) + x) + (0.724 * ((z + x)^2) + y) - (0.230 * ((x + y)^2) + z) + (1.350 * (x + y*z)^2) - (1.455 * (y + x*z)^2) + (0.018 * (z + x*y)^2) + (1.350 * ((x + y)*z)^2) - (1.456 * ((x + z)*y)^2) + (0.018 * ((y + z)*x)^2) + (1.350 * ((x + y)*z + x)^2) - (1.455 * ((x + z)*y + z)^2) + (0.017 * ((y + z)*x + z)^2) - (1.350 * (x + y*z)^2 + x) + (1.455 * (y + x*z)^2 + y) - (0.018 * (z + x*y)^2 + z) - (1.350 * ((x + y)*z)^2 + x) + (1.455 * ((x + z)*y)^2 + y) - (0.017 * ((y + z)*x)^2 + z) - (1.350 * ((x + y)*z + x)^2 + x) + (1.456 * ((x + z)*y + z)^2 + y) - (0.018 * ((y + z)*x + z)^2 + z)"

# user_input = "4.3*x^2 + 0.25*y + 8*(z*(x+y))"

# user_input = "(4*x^2) + (8*x^3) + 4*x*y*z"

# user_input = "x - (2*x^3) + (3*x^5) - (4*y*x^7)"

# user_input = "x*((3 - (4*x^2)*(x^2)) + 1)"

# #user_input = "4.5*(x + (x*y + z)) + y^2"
