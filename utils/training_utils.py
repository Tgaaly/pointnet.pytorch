from enum import Enum 

class print_colors(Enum):
    # For colors in displaying.
    blue = staticmethod(lambda x:'\033[94m' + x + '\033[0m')
    green = staticmethod(lambda x:'\033[92m' + x + '\033[0m')
