import json
import sys

if len(sys.argv) != 2:
    print("Usage: python count_xgb_nodes.py /eos/user/s/swaldych/smart_pix/labels/models/xgboost_trees.json")
    sys.exit(1)

json_file = sys.argv[1]

# Load trees
with open(json_file, "r") as f:
    trees = json.load(f)

def count_nodes(node):
    """Recursively count nodes in one tree."""
    if "children" not in node:
        return 1
    return 1 + sum(count_nodes(child) for child in node["children"])

num_trees = len(trees)

nodes_per_tree = []
total_nodes = 0

for i, tree in enumerate(trees):
    n = count_nodes(tree)
    nodes_per_tree.append(n)
    total_nodes += n
    print(f"Tree {i}: {n} nodes")

print("\n========================")
print("Summary")
print("========================")
print("Total trees:", num_trees)
print("Total nodes:", total_nodes)
print("Average nodes/tree:", total_nodes / num_trees)
print("Max nodes in one tree:", max(nodes_per_tree))
print("Min nodes in one tree:", min(nodes_per_tree))
