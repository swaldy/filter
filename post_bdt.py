import json
import sys


JSON_FILE = "/eos/user/s/swaldych/smart_pix/labels/models/xgboost_model_50x12P5x150_0fb_0P2thresh.json"
NUM_CLASS = 3

with open(JSON_FILE, "r") as f:
    data = json.load(f)

# Pull trees from the save_model JSON structure
trees = data["learner"]["gradient_booster"]["model"]["trees"]
num_trees = len(trees)

# best_iteration is stored as a string in attributes
attrs = data["learner"].get("attributes", {})
best_iter = int(attrs.get("best_iteration", "-1"))
rounds = best_iter + 1 if best_iter >= 0 else None

# Count nodes per tree in save_model format:
# Each node index corresponds to one entry in left_children/right_children arrays.
def nodes_in_tree(tree):
    if "left_children" in tree:
        return len(tree["left_children"])
    # fallback to other per-node arrays if needed
    for k in ["right_children", "split_indices", "split_conditions", "base_weights"]:
        if k in tree:
            return len(tree[k])
    raise KeyError(f"Can't infer node count from keys: {list(tree.keys())}")

nodes_per_tree = [nodes_in_tree(t) for t in trees]
total_nodes = sum(nodes_per_tree)

print("File:", JSON_FILE)
print("Total trees:", num_trees)

if rounds is not None:
    print("best_iteration:", best_iter)
    print("Boosting rounds:", rounds)
    print("num_class:", NUM_CLASS)
    print("Expected trees (rounds * num_class):", rounds * NUM_CLASS)

print("\nNode counts:")
print("Total nodes:", total_nodes)
print("Average nodes/tree:", total_nodes / num_trees)
print("Max nodes in one tree:", max(nodes_per_tree))
print("Min nodes in one tree:", min(nodes_per_tree))
