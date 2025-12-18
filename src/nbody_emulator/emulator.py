# emulator.py

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices("cpu")[0])
import jax.numpy as jnp
import equinox as eqx
import optax

class NNEmulator(eqx.Module):
    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear
    layer3: eqx.nn.Linear
    
    def __init__(self, key, input_size=2, hidden_size=64, output_size=3):
        key1, key2, key3 = jax.random.split(key, 3)
        
        # Input layer --> hidden layer 1 --> hidden layer 2 --> output
        self.layer1 = eqx.nn.Linear(input_size, hidden_size, key=key1)
        self.layer2 = eqx.nn.Linear(hidden_size, hidden_size, key=key2)
        self.layer3 = eqx.nn.Linear(hidden_size, output_size, key=key3)
        pass
    
    def __call__(self, x):
        # Hidden layer 1: linear --> ReLU
        x = self.layer1(x)
        x = jax.nn.relu(x)
        
        # Hidden layer 2: linear --> ReLU
        x = self.layer2(x)
        x = jax.nn.relu(x)
        
        # Output layer: linear (no activation for regression)
        x = self.layer3(x)
        
        return x

def mse_loss(model, x, y):
    """
    Compute mean squared error loss
    
    Args:
        model: NNEmulator object
        x: (N, 2) array of normalized inputs
        y: (N, 3) array of normalized outputs
    
    Returns:
        float: MSE loss
    """
    pred = jax.vmap(model)(x)
    
    return jnp.mean((pred - y) ** 2)

@eqx.filter_jit
def train_step(model, opt_state, optimizer, x, y):
    """
    Single step
    
    Args:
        model: NNEmulator
        opt_state: optimizer state
        optimizer: Optax optimizer
        x: batch of inputs (N, 2)
        y: batch of outputs (N, 3)
    
    Returns:
        new_model: updated model
        new_opt_state: updated optimizer state
        loss: current loss value
    """

    loss, grads = eqx.filter_value_and_grad(mse_loss)(model, x, y)
    
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    
    new_model = eqx.apply_updates(model, updates)
    
    return new_model, new_opt_state, loss

def train_single_model(key, x_train, y_train, learning_rate=1e-3, num_epochs=1000, print_every=100):
    """
    Train a single neural network emulator.
    
    Args:
        key: JAX random key
        x_train: (N, 2) normalized training inputs
        y_train: (N, 3) normalized training outputs
        learning_rate: Adam learning rate
        num_epochs: number of training epochs
        print_every: print loss every N epochs
    
    Returns:
        model: trained NNEmulator
        losses: list of training losses
    """
    model = NNEmulator(key)
    
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    losses = []
    
    print(f"Training single model...")
    print(f"  Input shape: {x_train.shape}")
    print(f"  Output shape: {y_train.shape}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {num_epochs}\n")
    
    for epoch in range(num_epochs):
        # Single training step
        model, opt_state, loss = train_step(model, opt_state, optimizer, x_train, y_train)
        losses.append(float(loss))
        
        # Print progress
        if epoch % print_every == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")
        
        # Check for NaN
        if jnp.isnan(loss):
            print(f"\nWARNING: Loss became NaN at epoch {epoch}!")
            print("Try reducing learning rate or checking your data normalization.")
            break
    
    print(f"\nTraining complete!")
    print(f"Final loss: {losses[-1]:.6f}")
    
    return model, losses

def train_ensemble(key, x_train, y_train, n_models=5, **train_kwargs):
    """
    Train multiple models with different initializations.
    
    Args:
        key: JAX random key
        x_train: (N, 2) normalized training inputs
        y_train: (N, 3) normalized training outputs
        n_models: number of ensemble members
        **train_kwargs: additional arguments for train_single_model
    
    Returns:
        models: list of trained NNEmulator instances
        all_losses: list of loss histories for each model
    """
    keys = jax.random.split(key, n_models)
    
    models = []
    all_losses = []
    
    print(f"Training ensemble of {n_models} models...\n")
    
    for i, model_key in enumerate(keys):
        print(f"=" * 60)
        print(f"Training model {i+1}/{n_models}")
        print(f"=" * 60)
        
        model, losses = train_single_model(model_key, x_train, y_train, **train_kwargs)
        
        models.append(model)
        all_losses.append(losses)
        
        print()
    
    return models, all_losses

def predict_ensemble(models, x):
    """
    Get ensemble predictions with uncertainty.
    
    Args:
        models: list of trained NNEmulator instances
        x: (N, 2) array of normalized inputs OR (2,) single input
    
    Returns:
        mean: (N, 3) or (3,) mean predictions
        std: (N, 3) or (3,) standard deviations
    """
    single_input = (x.ndim == 1)
    if single_input:
        x = x[None, :]
    
    predictions = []
    for model in models:
        pred = jax.vmap(model)(x)
        predictions.append(pred)
    
    predictions = jnp.array(predictions)
    
    mean = jnp.mean(predictions, axis=0)
    std = jnp.std(predictions, axis=0, ddof=1)

    if single_input:
        mean = mean[0]
        std = std[0]
    
    return mean, std