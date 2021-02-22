class Tree:
    def __init__(self, children=None, depth=0, val=0, x=0, y=0):
        self.val = val
        self.x = x
        self.y = y
        self.children = []
        self.depth = depth
        if children is not None:
            self.add_children_from_array(children)

    def add_children_from_array(self, array):
        for i in range(array.shape[0]):
            node = Tree(depth = self.depth + 1, val=array[i,0], x=array[i,1], y=array[i,2])
            self.add_child(node)
        
    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)
