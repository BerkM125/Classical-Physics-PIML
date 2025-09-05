# Author: Berkan Mertan
# Copyright (c) 2025 Berkan Mertan. All rights reserved.

# This program trains a PINN to model the angle of the pendulum
# from the vertical as a function of time

# It solves a standard diff equation modeling pendulum motion
# d²θ/dt² + (g/L)sin(θ) = 0
# Where:
#   θ is the angle of the pendulum from the vertical.
#   t is time.
#   g is acceleration due to gravity.
#   L is the length of the pendulum.

# External dependencies
import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
from scipy.integrate import solve_ivp

# Local modules
from models.lightweight_pendulum_pinn import PendulumPINN
from model_loader import ModelLoader
from plotter import Plotter

# These constants can be adjusted
g = 9.81
L = 2.0   # Length of the pendulum (m)
initial_angle = torch.tensor([np.pi / 4], requires_grad=True).float() # Start angle (radians)
initial_omega = torch.tensor([0.0], requires_grad=True).float()       # Initial angular velocity (rad/s)

# PINN loss function combines both data-based and physics-informed loss
def pinn_loss(model, t_data, t_physics, g, L):
    # The network must predict the correct angle and velocity at t=0.
    t_zero = torch.tensor([[0.0]], requires_grad=True).float()
    
    # Predict the initial angle and calculate its derivative (velocity)
    theta_pred_initial = model(t_zero)
    dtheta_dt_pred_initial = grad(theta_pred_initial, t_zero, torch.ones_like(theta_pred_initial), create_graph=True)[0]
    
    # MSE for initial conditions
    data_loss_angle = torch.mean((theta_pred_initial - initial_angle)**2)
    data_loss_omega = torch.mean((dtheta_dt_pred_initial - initial_omega)**2)
    data_loss = data_loss_angle + data_loss_omega
    
    # The network's output must satisfy the ODE for a range of time points.
    # Generate predicted angle for physics points
    theta_pred = model(t_physics)
    # Use autograd to get the first derivative, angular velocity
    dtheta_dt = grad(theta_pred, t_physics, torch.ones_like(theta_pred), create_graph=True)[0]
    # Use autograd again to get the 2nd derivative. angular acceleration
    d2theta_dt2 = grad(dtheta_dt, t_physics, torch.ones_like(dtheta_dt), create_graph=True)[0]

    # Calculate the residual of the pendulum's equation, thinking back to the eq
    physics_residual = d2theta_dt2 + (g / L) * torch.sin(theta_pred)
    
    # Our physics loss value will be the MSE of this residual
    physics_loss = torch.mean(physics_residual**2)
    
    # The total loss is the sum of the data and physics losses
    total_loss = data_loss + physics_loss
    return total_loss, data_loss, physics_loss

# Establish a training loop incorporating second order pendulum diff eq into the loss function
# This is important to compare model performance over different quantities of epochs
def PhysicsInformedTraining(epochs:int=50000, motion_duration:int=10.0):
    # Create the model and optimizer
    model = PendulumPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # For physics loss, we'll make a field of time points
    t_physics = torch.linspace(0, motion_duration, 100).view(-1, 1).requires_grad_(True)
    
    # The data points are just the initial conditions.
    t_data = torch.tensor([[0.0]], requires_grad=True).float()

    # Essentially we've just created a spatiotemporal dataset to feed to the network

    print(f"Initiating PINN training for {epochs} epochs...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Compute the loss
        loss, data_loss, physics_loss = pinn_loss(model, t_data, t_physics, g, L)
        
        # Backpropagate and update weights
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % (epochs/10) == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Total Loss: {loss.item():.6f}, "
                  f"Data Loss: {data_loss.item():.6f}, Physics Loss: {physics_loss.item():.6f}")
    return model

# Also train a standard model
def StandardLossMinimizationTraining(epochs:int=50000):
    pass 

def PlotModelPredictions(motion_time:float=10.0, model=None):
    """
    Plots predictions of the provided PINN along with
    the outputs of a known functon side by side

    motion_time: float, full time duration of pendulum motion to track (optional)
    model: nn.Module, extension of nn.Module class containing PINN (required)
    """

    # Simple function representing a known rads(time) function
    def pendulum_ode(t, y):
        theta, omega = y
        dtheta_dt = omega
        domega_dt = -(g / L) * np.sin(theta)
        return [dtheta_dt, domega_dt]

    # Solve the ODE using a standard library
    sol = solve_ivp(pendulum_ode, [0, motion_time], [initial_angle.item(), initial_omega.item()], 
                    dense_output=True, rtol=1e-8, atol=1e-8)
    
    # Generate time points for plotting
    t_test = np.linspace(0, motion_time, 500)
    theta_sol = sol.sol(t_test)[0]

    # Get the PINN's predictions on the same time points
    with torch.no_grad():
        t_test_tensor = torch.tensor(t_test).view(-1, 1).float()
        theta_pinn = model(t_test_tensor).numpy()    

    plotman = Plotter(xFields=[t_test, t_test],
                      yFields=[theta_sol, theta_pinn],
                      xLabel="Time (Seconds)",
                      yLabel="Angle (Radians)",
                      title="Angle (Radians) from the Vertical over Time (S)",)
    plotman.plot()

# Main func 
if __name__ == '__main__':
    # We can control the number of epochs to improve accuracy.
    epochs = int(input("Enter training epochs limit: "))

    pendulum_model = PhysicsInformedTraining(epochs=epochs,
                                             motion_duration=5.0)
    
    #Check "out" folder for output pytorch models, feel free to load from there after training
    ModelLoader.save_model(model=pendulum_model, path="../out/pendulum_pinn.pth")

    #pendulum_model = ModelLoader.load_model(path="../out/pendulum_pinn.pth")

    print("\nTraining complete! Generating plot...")

    PlotModelPredictions(motion_time=4.0,
                         model=pendulum_model)
    
    print("\nPlotting complete, exiting...")
