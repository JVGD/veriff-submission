from src.stnmodel import STNModel
from src.coordconv import CoordConv


class CoordConvModel(STNModel):
    """CoordConv Model

    Same as base model STNModel but replacing
    conv by coord conv the rest is the same
    """
    def __init__(self, optimizer: dict) -> None:
        """Init CoordConvModel from STNModel setup

        Args:
            optimizer (dict): Dict with optimizer conf
        """
        # Init base class
        super().__init__(optimizer)

        # Overwriting conv with CoordConv
        self.conv1 = CoordConv(1, 10, kernel_size=5)
        self.conv2 = CoordConv(10, 20, kernel_size=5)
