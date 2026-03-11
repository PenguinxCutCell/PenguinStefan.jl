module PenguinStefan

using LinearAlgebra
using SparseArrays
using StaticArrays
using LinearSolve

using CartesianGrids
using CartesianGeometry
using CartesianOperators
using GlobalHeightFunctions
using PenguinBCs
using PenguinSolverCore
using PenguinDiffusion
using LevelSetMethods
using SpecialFunctions
import PenguinSolverCore: solve!

export AbstractInterfaceRep, LevelSetRep, GlobalHFRep
export StefanParams, StefanOptions
export StefanMonoProblem, StefanDiphProblem
export StefanMonoState, StefanDiphState
export build_solver, step!, solve!
export coupling_mode, coupled_step!
export stefan_lambda_neumann, neumann_1d_similarity, manufactured_planar_1d, frank_disk_exact
export active_mask, phase_volume, interface_position_from_volume, radius_from_volume
export temperature_error_norms, stefan_residual_metrics, fit_order

include("interface/abstract.jl")
include("types.jl")
include("validation/exact_solutions.jl")
include("validation/metrics.jl")
include("interface/levelset_rep.jl")
include("interface/ghf_rep.jl")
include("interface/predictors.jl")
include("interface/gibbs_thomson.jl")
include("moving_diffusion/mono.jl")
include("moving_diffusion/diph.jl")
include("stefan_speed.jl")
include("stepper_mono.jl")
include("stepper_diph.jl")
include("interface/ghf_coupling.jl")
include("io.jl")

end # module
