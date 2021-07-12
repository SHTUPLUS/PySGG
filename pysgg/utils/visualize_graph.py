import os
import random

import ipdb
import torch
from graphviz import Digraph


def visual_computation_graph(var, params, output_dir, graph_name='network'):
    """ Produces Graphviz representation of PyTorch autograd graph.

    Blue nodes are trainable Variables (weights, bias).
    Orange node are saved tensors for the backward pass.

    Args:
        var: output Variable
        params: list of (name, Parameters)
    """

    param_map = {id(v): k for k, v in params}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')

    comp_graph = Digraph(filename=os.path.join(output_dir, graph_name),
                          format='pdf',
                          node_attr=node_attr,
                          graph_attr=dict(size="256,512"))
    seen = set()



    def get_color():
        pallet = ['#8B0000', "#FF8C00", "#556B2F", "#8FBC8F", "#2F4F4F", "#4682B4",
                  "#191970", "#8A2BE2", "#C71585", "#000000", "#808080"]

        idx = random.randint(0, len(pallet)-1)
        return pallet[idx]


    def add_nodes(var):
        if var not in seen:

            node_id = str(id(var))

            if torch.is_tensor(var):
                node_label = "saved tensor\n{}".format(tuple(var.size()))
                comp_graph.node(node_id, node_label, fillcolor='orange')

            elif hasattr(var, 'variable'):
                # weights
                variable_name = param_map.get(id(var.variable))
                variable_size = tuple(var.variable.size())
                node_name = "{}\n{}".format(variable_name, variable_size)
                comp_graph.node(node_id, node_name, fillcolor='lightblue')

            else:
                # operation
                node_label = type(var).__name__.replace('Backward', '')
                comp_graph.node(node_id, node_label)

            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        comp_graph.edge(str(id(u[0])), str(id(var)), color=get_color())
                        add_nodes(u[0])

            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    comp_graph.edge(str(id(t)), str(id(var)), color=get_color())
                    add_nodes(t)

    add_nodes(var.grad_fn)

    return comp_graph
