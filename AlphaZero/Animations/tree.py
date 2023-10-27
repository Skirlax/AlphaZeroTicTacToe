from manimlib import Circle,Text,Scene,VGroup

class TreeNode(Circle):
    def __init__(self, named_values: dict, **kwargs):
        super().__init__(**kwargs)
        self.named_values = named_values
        self.make_values()

    def make_values(self):
        for key, value in self.named_values.items():
            self.add(Text(f"{key}: {value}\n"))


class SearchTree(Scene):
    def construct(self) -> None:
        pass

    def draw_tree(self):
        pass

    def generate_sample_nodes(self,depth: int):
        nodes = VGroup()
        for i in range(depth):
            pass



if __name__ == "__main__":
    tree = SearchTree()
    tree.render()


