using DifferentialEquations, DelimitedFiles, Plots, LinearAlgebra

const e0 = 2.5
const v0 = 6
const v0_p2 = 1
const r = 0.56

sigma(x, v0) = 2 * e0 / (1 + exp(r * (v0 - x)))

const A_AMPA, A_GABAs, A_GABAf = 3.25, -22, -30
const a_AMPA, a_GABAs, a_GABAf = 100, 50, 220

const C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12 = 108, 33.7, 1, 135, 33.75, 70, 550, 1, 200, 100, 80, 200, 30

const p1, p2 = 200, 150 #p1 is noisy in the original model, produced by N(200,30)
const I1, I2 = (A_AMPA/a_AMPA) * p1, (A_AMPA/a_AMPA) * p2



function LaNMM(dx, x, p, t)
    # Model implementation
    dx[1] = x[6]
    dx[6] = A_AMPA*a_AMPA*(sigma(C10*x[4]+C1*x[3]+C0*x[2]+I1, v0))-2*a_AMPA*x[6]-a_AMPA^2*x[1]
    dx[2] = x[7]
    dx[7] = A_AMPA*a_AMPA*(sigma(C3*x[1], v0))-2*a_AMPA*x[7]-a_AMPA^2*x[2]
    dx[3] = x[8]
    dx[8] =A_GABAs*a_GABAs*sigma(C4*x[1], v0) -2*a_GABAs*x[8] -a_GABAs^2*x[8]
    dx[4] = x[9]
    dx[9] = A_AMPA*a_AMPA*sigma(C11*x[1]+ C5*x[4] + C6*x[5]+ I2, v0_p2) -2*a_AMPA*x[9] -a_AMPA^2*x[4]
    dx[5] = x[10]
    dx[10] = A_GABAf*a_GABAf*sigma(C12*x[1] + C8*x[4] + C9*x[5], v0) -2*a_GABAf*x[10] - a_GABAf^2*x[5]
end



# Define the initial conditions and parameters
u0 = [.2, .2, .2, .2, .2, 0, 0, 0, 0, 0]
tspan = (0.0, 10.0)
p = []

# Solve the differential equation using the `solve` function
prob = ODEProblem(myODE, u0, tspan, p)
sol = solve(prob)





