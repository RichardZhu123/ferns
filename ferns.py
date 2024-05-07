import numpy as np
import sys

## Create red black tree datastructure

class VectorNode:
    def __init__(self, vector):
        self.vector = vector
        self.left = None
        self.right = None
        self.parent = None
        self.isRed = True
        self.depth = 0

class VectorTree:
    def __init__(self):
        self.root = VectorNode(None)
        self.root.isRed = False
        self.max_depth = 0

    def insert(self, vector):
        if self.root is None:
            self.root = VectorNode(vector)
            self.root.isRed = False
        else:
            node = self._insert(vector, self.root)
            # self._fix_tree(node)

    def _insert(self, vector, node):
        if node.left is None:
            node.left = VectorNode(vector)
            node.left.parent = node
            node.left.isRed = True
            node.left.depth = node.depth + 1
            self.max_depth = max(node.left.depth, self.max_depth)
            return node.left
        elif node.right is None:
            node.right = VectorNode(vector)
            node.right.parent = node
            node.right.isRed = True
            node.right.depth = node.depth + 1
            self.max_depth = max(node.right.depth, self.max_depth)
            return node.right
        elif self.compare(vector, node.left.vector, node.right.vector)[0] < 0:
            return self._insert(vector, node.left)
        else:
            return self._insert(vector, node.right)

    def retrieve_nearest(self, query):
        """
        Retrieve the nearest (by cosine similarity) neighbor to the given vector.
        """
        # distances = SortedDict()
        # mip is maximum inner product
        mip, mip_vector = sys.float_info.max, None
        curr = None
        queue = [self.root] # queue of VectorNodes to visit
        num_visited = 0

        while queue:
            curr = queue.pop(0)
            num_visited += 1
            if curr is not self.root:
                mip, mip_vector = self.closest(mip, mip_vector, curr.vector, query)
            if curr.left and curr.right:
                lr_dot, can_prune = self.compare(query, curr.left.vector, curr.right.vector)
                if lr_dot <= 0:
                    queue.append(curr.left)
                    if not can_prune: 
                        queue.append(curr.right)
                else:
                    queue.append(curr.right)
                    if not can_prune: 
                        queue.append(curr.left)
            else:
                if curr.left:
                    queue.append(curr.left)
                elif curr.right:
                    queue.append(curr.right)

        return mip_vector, num_visited
    
    def closest(self, min_dist, nn_vector, vector, query):
        vq_dist = self.euclidean_distance(vector, query)
        if vq_dist < min_dist:
            return vq_dist, vector
        return min_dist, nn_vector
    
    def cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def euclidean_distance(self, v1, v2):
        return np.linalg.norm(v1 - v2)
        
    def compare(self, query, v1, v2):
        """
            if v1 is closer to query than v2, compare() < 0
        """
        v1v2_dist = self.euclidean_distance(v1, v2) ** 2
        if v1v2_dist == 0:
            return 0, True
        proportion_v2_euc = np.dot(v1-v2, query-v2) / v1v2_dist
        proportion_v1_euc = np.dot(v2-v1, query-v1) / v1v2_dist
        return 0.5 - proportion_v2_euc, True
    
