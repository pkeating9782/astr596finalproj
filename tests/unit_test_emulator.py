# unit_test_emulator.py
import jax
import jax.numpy as jnp
from src.nbody_emulator import emulator, utils

def test_normalization_roundtrip():
    """Test that normalize -> denormalize recovers original data."""
    data = jnp.array([[0.5, 50.0], [1.0, 125.0], [1.5, 200.0]])
    
    normalizer = utils.Normalizer()
    normalizer.fit(data)
    
    normalized = normalizer.normalize(data)
    recovered = normalizer.denormalize(normalized)
    
    assert jnp.allclose(data, recovered, atol=1e-6), "Normalization roundtrip failed"
    print("✓ Normalization roundtrip test passed")

def test_model_output_shape():
    """Test that model produces correct output shape."""
    key = jax.random.PRNGKey(42)
    model = emulator.NNEmulator(key)
    
    # Single input
    x = jnp.array([0.0, 0.0])  # normalized
    output = model(x)
    assert output.shape == (3,), f"Expected shape (3,), got {output.shape}"
    
    # Batch input
    x_batch = jnp.zeros((5, 2))
    output_batch = jax.vmap(model)(x_batch)
    assert output_batch.shape == (5, 3), f"Expected shape (5, 3), got {output_batch.shape}"
    
    print("✓ Model output shape test passed")

def test_no_nan_in_predictions():
    """Test that model doesn't produce NaN values."""
    key = jax.random.PRNGKey(42)
    model = emulator.NNEmulator(key)
    
    x = jnp.array([0.0, 0.0])
    output = model(x)
    
    assert not jnp.any(jnp.isnan(output)), "Model produced NaN values"
    print("✓ No NaN test passed")

if __name__ == "__main__":
    print("Running basic sanity tests...\n")
    test_normalization_roundtrip()
    test_model_output_shape()
    test_no_nan_in_predictions()
    print("\n✓ All tests passed!")