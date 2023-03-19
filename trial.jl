
using DifferentialEquations

function myODE(du, u, p, t)
    # Define the differential equation: du/dt = -u
    du[1] = -u[1]
end

# Define the initial conditions and parameters
u0 = [1.0]
tspan = (0.0, 10.0)
p = []

# Solve the differential equation using the `solve` function
prob = ODEProblem(myODE, u0, tspan, p)
sol = solve(prob)

# Plot the solution
using Plots
plot(sol)
print("EHLAMADò")