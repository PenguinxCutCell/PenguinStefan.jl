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

@testset "Speed extension seeds opposite interface side in 2D" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (65, 65))
    s0 = 0.31
    rep = LevelSetRep(grid, (x, y) -> x - s0)

    ϕ = PenguinStefan.phi_values(rep)
    v = zeros(Float64, size(ϕ))
    frozen = falses(size(ϕ))

    # Mimic the Stefan assignment: values provided only on one side of the interface.
    for I in CartesianIndices(ϕ)
        i, j = Tuple(I)
        if ϕ[i, j] <= 0.0
            ip = min(i + 1, size(ϕ, 1))
            if ϕ[ip, j] > 0.0
                v[i, j] = 0.2
                frozen[i, j] = true
            end
        end
    end

    PenguinStefan.extend_speed!(
        rep,
        v;
        frozen=frozen,
        nb_iters=12,
        cfl=0.45,
        interface_band=4.0,
    )

    # On the centerline, check that interface interpolation recovers the imposed speed.
    j = Int(cld(size(ϕ, 2), 2))
    x = CartesianGrids.grid1d(grid, 1)
    i0 = findfirst(xi -> xi >= s0, x)
    @test i0 !== nothing
    i = i0 - 1
    θ = abs(ϕ[i, j]) / (abs(ϕ[i, j]) + abs(ϕ[i + 1, j]))
    v_if = (1 - θ) * v[i, j] + θ * v[i + 1, j]
    @test isapprox(v_if, 0.2; atol=2e-2)
end
