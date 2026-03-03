using Test
using CartesianGrids
using PenguinStefan

@testset "Speed extension keeps frozen values" begin
    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (41, 41))
    rep = LevelSetRep(grid, (x, y) -> sqrt(x^2 + y^2) - 0.45)

    ϕ = PenguinStefan.phi_values(rep)
    v = zeros(Float64, size(ϕ))
    frozen = falses(size(ϕ))

    for j in axes(ϕ, 2), i in axes(ϕ, 1)
        if abs(ϕ[i, j]) <= 0.06
            v[i, j] = sin(pi * i / size(ϕ, 1))
            frozen[i, j] = true
        end
    end

    nz0 = count(abs.(v) .> 1e-12)
    vfrozen0 = copy(v[frozen])

    PenguinStefan.extend_speed!(
        rep,
        v;
        frozen=frozen,
        nb_iters=12,
        cfl=0.45,
        interface_band=3.0,
    )

    @test maximum(abs.(v[frozen] .- vfrozen0)) < 1e-12
    @test count(abs.(v) .> 1e-12) > nz0
end
