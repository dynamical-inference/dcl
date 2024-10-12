import torch
from dcl.models.encoder import IdentityEncoder
from dcl.models.encoder import LinearEncoderModel
from dcl.models.encoder import MLP
from dcl.models.encoder import Offset1ModelMLP


def test_model_weight_initialization():
    """Test that models with same seed have same weights and different seeds have different weights."""
    # Test dimensions
    input_dim = 5
    output_dim = 3
    hidden_dim = 128
    num_layers = 3

    # Test MLP model
    mlp1_same = MLP(input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    seed=42)
    mlp2_same = MLP(input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    seed=42)
    mlp_diff = MLP(input_dim=input_dim,
                   output_dim=output_dim,
                   hidden_dim=hidden_dim,
                   num_layers=num_layers,
                   seed=43)

    # Test that MLP models with same seed have identical weights
    for (name1, param1), (name2, param2) in zip(mlp1_same.named_parameters(),
                                                mlp2_same.named_parameters()):
        assert torch.allclose(
            param1, param2
        ), f"MLP parameters {name1} differ between models with same seed"

    # Test that MLP models with different seeds have different weights
    for (name1, param1), (name2, param2) in zip(mlp1_same.named_parameters(),
                                                mlp_diff.named_parameters()):
        assert not torch.allclose(
            param1, param2
        ), f"MLP parameters {name1} are identical between models with different seeds"

    # Test LinearEncoderModel
    linear1_same = LinearEncoderModel(input_dim=input_dim,
                                      output_dim=output_dim,
                                      seed=42)
    linear2_same = LinearEncoderModel(input_dim=input_dim,
                                      output_dim=output_dim,
                                      seed=42)
    linear_diff = LinearEncoderModel(input_dim=input_dim,
                                     output_dim=output_dim,
                                     seed=43)

    # Test that linear models with same seed have identical weights
    for (name1, param1), (name2,
                          param2) in zip(linear1_same.named_parameters(),
                                         linear2_same.named_parameters()):
        assert torch.allclose(
            param1, param2
        ), f"Linear parameters {name1} differ between models with same seed"

    # Test that linear models with different seeds have different weights
    for (name1, param1), (name2, param2) in zip(linear1_same.named_parameters(),
                                                linear_diff.named_parameters()):
        assert not torch.allclose(
            param1, param2
        ), f"Linear parameters {name1} are identical between models with different seeds"

    # Test Offset1ModelMLP
    offset1_same = Offset1ModelMLP(input_dim=input_dim,
                                   output_dim=output_dim,
                                   seed=42)
    offset2_same = Offset1ModelMLP(input_dim=input_dim,
                                   output_dim=output_dim,
                                   seed=42)
    offset_diff = Offset1ModelMLP(input_dim=input_dim,
                                  output_dim=output_dim,
                                  seed=43)

    # Test that offset models with same seed have identical weights
    for (name1, param1), (name2,
                          param2) in zip(offset1_same.named_parameters(),
                                         offset2_same.named_parameters()):
        assert torch.allclose(
            param1, param2
        ), f"Offset parameters {name1} differ between models with same seed"

    # Test that offset models with different seeds have different weights
    for (name1, param1), (name2, param2) in zip(offset1_same.named_parameters(),
                                                offset_diff.named_parameters()):
        assert not torch.allclose(
            param1, param2
        ), f"Offset parameters {name1} are identical between models with different seeds"

    # Test IdentityEncoder
    identity1_same = IdentityEncoder(input_dim=input_dim,
                                     output_dim=output_dim,
                                     seed=42)
    identity2_same = IdentityEncoder(input_dim=input_dim,
                                     output_dim=output_dim,
                                     seed=42)
    identity_diff = IdentityEncoder(input_dim=input_dim,
                                    output_dim=output_dim,
                                    seed=43)

    # Test that identity models with same seed have identical weights
    for (name1, param1), (name2,
                          param2) in zip(identity1_same.named_parameters(),
                                         identity2_same.named_parameters()):
        assert torch.allclose(
            param1, param2
        ), f"Identity parameters {name1} differ between models with same seed"

    # Test that identity models with different seeds have identical weights
    for (name1, param1), (name2,
                          param2) in zip(identity1_same.named_parameters(),
                                         identity_diff.named_parameters()):
        assert torch.allclose(
            param1, param2
        ), f"Identity parameters {name1} differ between models with different seeds"
