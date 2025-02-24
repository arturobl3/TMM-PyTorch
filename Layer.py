from typing import List, Tuple, Literal, Dict
from collections import defaultdict
from material import BaseMaterial
import torch 

# Define a literal type for layer classification.
LayerType = Literal["coherent", "substrate", "environment"]

class BaseLayer:
    """
    BaseLayer represents a generic layer in a material stack or optical system.
    
    This class encapsulates the fundamental properties of a layer including
    its material, thickness, and type. It is intended to be extended by more
    specialized layer classes as needed.
    
    Attributes:
        material (BaseMaterial): The material associated with the layer.
        thickness (torch.Tensor): The physical thickness of the layer.
        type (LayerType): A literal string indicating the layer type, which
            can be one of the following:
                - "coherent": Layers that exhibit coherent interference effects.
                - "substrate": Layers that serve as a substrate in the structure.
                - "environment": Layers representing the surrounding environment.
    """

    def __init__(self,
                 material: BaseMaterial,
                 thickness: torch.Tensor,
                 LayerType: LayerType
                 ) -> None:
        """
        Initialize a BaseLayer instance with the specified material, thickness,
        and layer type.
        
        Args:
            material (BaseMaterial): An instance of BaseMaterial representing
                the layer's material properties.
            thickness (torch.Tensor): A tensor representing the thickness of the
                layer. The unit of thickness should be consistent with the
                material model being used.
            LayerType (LayerType): A literal value specifying the layer type.
                Accepted values are "coherent", "substrate", or "environment".
        """
        self.material = material
        self.thickness = thickness
        self.type = LayerType
