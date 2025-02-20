from typing import List, Tuple, Literal, Dict
from collections import defaultdict
from Material import BaseMaterial
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





class LayerStructure:
    def __init__(self):
        # This list maintains the global insertion order of all layers.
        self._layers: List[BaseLayer] = []
        # This dictionary maps each layer type to a list of layer references.
        self._layers_by_type: Dict[LayerType, List[BaseLayer]] = defaultdict(list)

    def add_layer(self, layer: BaseLayer) -> None:
        """
        Adds a layer to the collection.
        Assumes each layer has an attribute 'layer_type' (e.g., "material", "substrate").
        """
        self._layers.append(layer)
        self._layers_by_type[layer.layer_type].append(layer)

    def get_layers(self) -> List[BaseLayer]:
        """
        Returns all layers in their insertion order.
        """
        return self._layers

    def get_layers_by_type(self, layer_type: BaseLayer) -> List[BaseLayer]:
        """
        Returns a list of layers that match the specified type.
        """
        return self._layers_by_type.get(layer_type, [])

    def remove_layers_by_type(self, layer_type: BaseLayer) -> None:
        """
        Removes all layers of a given type.
        This operation updates both the ordered list and the dictionary.
        """
        # Filter the main list to remove layers of the given type.
        self._layers = [layer for layer in self._layers if layer.layer_type != layer_type]
        # Remove the entry from the dictionary.
        if layer_type in self._layers_by_type:
            del self._layers_by_type[layer_type]

    def __repr__(self) -> str:
        return f"OrderedLayerCollection(layers={self._layers})"


