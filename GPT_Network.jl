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

const epsilon = 10
const N = 90

function read_W(data_address)
    data = readdlm(data_address, skipstart=1)
    W = zeros(N, N)
    for row in data
        i, j, w = row
        W[Int(i), Int(j)] = w
    end
    return W
end

W = read_W("Data/data.dat")
X0 = hcat(fill(0.2, N, 5), zeros(N, 5))
dx = zeros(N, 10)

function Network_LaNMM(x, t)
    x = reshape(x, (N, 10))
    ext_p1 = epsilon * W * x[:, 1]
    for i in 1:N
        dx[i, 0] = x[i, 5] # P1
        dx[i, 5] = A_AMPA * a_AMPA * (sigma(C10*x[i,3]+C1*x[i,2]+C0*x[i,1]+I1 +ext_p1[i], v0)) - 2 * a_AMPA * x[i, 5] - a_AMPA^2 * x[i, 5]
        

        dx[i, 1] = x[i, 6] # SS population
        dx[i, 6] = A_AMPA * a_AMPA * (sigma(C3 * x[i, 1], v0)) - 2 * a_AMPA * x[i, 6] - a_AMPA^2 * x[i, 6]
        
        dx[i, 2] = x[i, 7] # SST population
        dx[i, 7] = A_GABAs * a_GABAs * sigma(C4 * x[i, 1], v0) - 2 * a_GABAs * x[i, 7] - a_GABAs^2 * x[i, 7]

        dx[i, 3] = x[i, 8] # P2 population
        dx[i, 8] = A_AMPA * a_AMPA * sigma(C11 * x[i, 1] + C5 * x[i, 3] + C6 * x[i, 4] + I2, v0_p2) - 2 * a_AMPA * x[i, 8] - a_AMPA^2 * x[i, 8]
        
        dx[i, 4] = x[i, 9] # PV population
        dx[i, 9] = A_AMPA * a_AMPA * sigma(C12*x[i,0] + C8*x[i,3] + C9*x[i,4], v0) - 2 * a_AMPA * x[i, 9] - a_AMPA^2 * x[i, 9]

    
