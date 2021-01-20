import CartesianGrids: curl!, Laplacian
import LinearAlgebra: Diagonal

export VortexModel, computeψ, computew, computew!, computevortexvelocities, computeregularizationmatrix, getstrengths, getpositions, setvortexpositions!, getvortexpositions, setvortices!, pushvortices!, computeimpulse, solvesystem, solvesystem!

mutable struct VortexModel{Nb,Ne,TU,TF}
    g::PhysicalGrid
    vortices::VortexList
    bodies::BodyList
    edges::Vector{Int}
    system::Union{PotentialFlowSystem,Laplacian}

    _nodedata::TU
    _bodydata::TF
    _w::TU
    _ψb::TF
    _Rvmat::Union{Nothing,RegularizationMatrix{TU}}
    _Evmat::Union{Nothing,InterpolationMatrix{TU}}
    _Rbmat::RegularizationMatrix{TU}
    _Ebmat::InterpolationMatrix{TU}
end

function VortexModel(g::PhysicalGrid; bodies::Union{Body,Vector{<:Body},BodyList}=BodyList(),  vortices::Union{Vortex,Vector{<:Vortex},VortexList}=VortexList(), edges::Vector{<:Integer}=Int[])

    # Ensure that bodies and vortices are of type BodyList and VortexList
    if bodies isa Body
        bodies = BodyList([bodies])
    elseif bodies isa Vector{<:Body}
        bodies = BodyList(bodies)
    end

    Nb = length(bodies)
    Ne = length(edges)

    # Initialize data structures for internal use
    _nodedata = Nodes(Dual,size(g))
    _bodydata = ScalarData(length(collect(bodies)[1]))
    _w = Nodes(Dual,size(g))
    _ψb = ScalarData(length(collect(bodies)[1]))

    # Create basic unregularized saddle point system S
    L = plan_laplacian(size(_nodedata),with_inverse=true)
    regop = Regularize(VectorData(collect(bodies)), cellsize(g), I0=origin(g), issymmetric=true, ddftype=CartesianGrids.Yang3)
    _Rbmat,_ = RegularizationMatrix(regop,_bodydata,_nodedata)
    _Ebmat = InterpolationMatrix(regop,_nodedata,_bodydata)
    S = SaddleSystem(L,_Ebmat,_Rbmat,SaddleVector(_nodedata,_bodydata))

    if isempty(edges) # Unregularized potential flow system with bodies
        _TF_ones = zeros(length(collect(bodies)[1]),Nb)
        for i in 1:Nb
            _TF_ones[getrange(bodies,i),i] .= 1;
        end
        system = PotentialFlowSystem(S,_TF_ones)
    else # Regularized potential flow system
        _nodedata .= 0
        _bodydata .= 1
        f₀ = constraint(S\SaddleVector(_nodedata,_bodydata));

        Df₀ = Diagonal(f₀);
        R̃bmat = deepcopy(_Rbmat);
        R̃bmat.M .= R̃bmat.M*Df₀;
        S̃ = SaddleSystem(L,_Ebmat,R̃bmat,SaddleVector(_nodedata,_bodydata))

        e_kvec = [BodyUnitVector(bodies[1],k) for k in edges]
        d_kvec = typeof(_nodedata)[]

        system = PotentialFlowSystem(S̃,f₀,e_kvec,d_kvec)
    end

    vortexmodel =  VortexModel{Nb,Ne,typeof(_nodedata),typeof(_bodydata)}(g, VortexList(), bodies, edges, system, _nodedata, _bodydata, _w, _ψb, nothing, nothing, _Rbmat, _Ebmat)

    setvortices!(vortexmodel, vortices)

    return vortexmodel
end

function issteady(vortexmodel::VortexModel)
    if isempty(vortexmodel.vortices)
        return true
    else
        return false
    end
end

function isregularized(vortexmodel::VortexModel{Nb,Ne}) where {Nb,Ne}
    if Ne > 0
        return true
    else
        return false
    end
end

function setvortices!(vortexmodel::VortexModel{Nb,Ne,TU,TF}, vortices::Union{Vortex,Vector{<:Vortex},VortexList}=VortexList(); computeregularization=true) where {Nb,Ne,TU,TF}

    vortices = VortexList(deepcopy(vortices))

    vortexmodel.vortices = vortices

    if computeregularization updatevorticesregularization(vortexmodel) end
end

function pushvortices!(vortexmodel::VortexModel{Nb,Ne,TU,TF}, vortices...; computeregularization=true) where {Nb,Ne,TU,TF}

    push!(vortexmodel.vortices.list,vortices...)

    if computeregularization updatevorticesregularization(vortexmodel) end
end

function setvortexpositions!(vortexmodel::VortexModel{Nb,Ne,TU,TF}, X_vortices::VectorData{Nv}; computeregularization=true) where {Nb,Ne,TU,TF,Nv}

    @assert Nv == length(vortexmodel.vortices)

    setpositions!(vortexmodel.vortices,X_vortices.u,X_vortices.v)

    if computeregularization updatevorticesregularization(vortexmodel) end
end

function getvortexpositions(vortexmodel::VortexModel{Nb,Ne,TU,TF}) where {Nb,Ne,TU,TF}

    return getpositions(vortexmodel.vortices)
end

function updatevorticesregularization(vortexmodel::VortexModel{Nb,Ne,TU,TF}) where {Nb,Ne,TU,TF}

    @unpack g, vortices, _nodedata, system = vortexmodel

    vortexmodel._Rvmat, vortexmodel._Evmat = computeregularizationmatrix(vortexmodel.g,getpositions(vortices),getstrengths(vortices),vortexmodel._nodedata)

    if isregularized(vortexmodel) && !isempty(vortices)
        d_kvec = computed_kvec(vortexmodel,collect(length(vortices)-(Ne-1):length(vortices)))
        setd_kvec!(system, d_kvec)
    end
end

function computeregularizationmatrix(g::PhysicalGrid, X::VectorData{N}, f::ScalarData{N}, s::Nodes) where {N}

    regop = Regularize(X, cellsize(g), I0=origin(g), ddftype = CartesianGrids.M4prime, issymmetric=true)
    Rmat,_ = RegularizationMatrix(regop, f, s)
    Emat = InterpolationMatrix(regop, s, f)

    return Rmat, Emat
end

function computed_kvec(vortexmodel::VortexModel{Nb,Ne,TU,TF}, indices::Vector{Int}) where {Nb,Ne,TU,TF}
    Γ = getstrengths(vortexmodel.vortices)
    d_kvec = TU[]
    for k in indices
        Γ .= 0
        Γ[k] = 1
        push!(d_kvec,vortexmodel._Rvmat*Γ)
    end
    return d_kvec
end

"""
    computevortexvelocities(vortexmodel::VortexModel, ψ::TU)

Returns the flow velocity as `VectorData` at the locations of the vortices stored in `vortexmodel` associated with the discrete vector potential field `ψ`.
"""
function computevortexvelocities(vortexmodel::VortexModel{Nb,Ne,TU,TF}, ψ::TU) where {Nb,Ne,TU,TF}

    @unpack g, vortices, _nodedata, _bodydata, _Evmat = vortexmodel

    Ẋ_vortices = VectorData(length(vortices))
    # Velocity is the curl of the vector potential
    # The discrete curl operator requires dividing by the cellsize to account for the grid spacing
    qedges = curl(ψ)/cellsize(g) # TODO: create _edgesdata to avoid allocating memory in this function

    # For consistent interpolation, first interpolate the velocity to the nodes and use _Evmat to interpolate from the nodes to the vortices
    grid_interpolate!(_nodedata,qedges.u);
    Ẋ_vortices.u .= _Evmat*_nodedata
    grid_interpolate!(_nodedata,qedges.v);
    Ẋ_vortices.v .= _Evmat*_nodedata

    return Ẋ_vortices
end

"""
    computevortexvelocities(vortexmodel::VortexModel, w::TU; kwargs...)

Returns the flow velocity as `VectorData` at the locations of the vortices stored in `vortexmodel` due to the vorticity field `w` and accounting for bodies in 'vortexmodel' and conditions in 'kwargs'.
"""
function computevortexvelocities(vortexmodel::VortexModel{Nb,Ne,TU,TF}; kwargs...) where {Nb,Ne,TU,TF}

    @unpack g, vortices, _nodedata, _bodydata, _Evmat = vortexmodel

    # The strengths of the Ne last vortices will be calculated in solvesystem and should be set to zero before computing the vorticity field such that they are not included in w
    for k in 1:Ne
        vortexmodel.vortices[end-Ne+k].Γ = 0.0
    end

    computew!(vortexmodel._w,vortexmodel)

    sol = solvesystem!(_nodedata, _bodydata, vortexmodel, vortexmodel._w; kwargs...)

    for k in 1:Ne
        vortices[end-Ne+k].Γ = sol.δΓ_kvec[k]
    end

    Ẋ_vortices = computevortexvelocities(vortexmodel,sol.ψ)

    return Ẋ_vortices
end

function computevortexvelocities(vortexmodel::VortexModel{Nb,Ne,TU,TF}, X_vortices::VectorData{Nv}; kwargs...) where {Nb,Ne,TU,TF,Nv}

    @assert Nv == length(vortexmodel.vortices)

    setvortexpositions!(vortexmodel,X_vortices)
    Ẋ_vortices = computevortexvelocities(vortexmodel; kwargs...)

    return Ẋ_vortices
end

"""
    computew!(w::Nodes,vortexmodel::VortexModel)

Computes the vorticity field `w` associated with the vortices stored in `vortexmodel` on the physical grid.
"""
function computew!(wphysical::TU, vortexmodel::VortexModel{Nb,Ne,TU,TF})::TU where {Nb,Ne,TU,TF}

    @unpack g, vortices, _Rvmat = vortexmodel

    if isempty(vortices)
        wphysical .= 0.0
        return wphysical
    end

    Γ = getstrengths(vortices)
    wphysical .= _Rvmat*Γ/cellsize(g)^2 # Divide by the Δx² to ensure that ∫wdA = ΣΓ

    return wphysical
end

function computew(vortexmodel::VortexModel{Nb,Ne,TU,TF})::TU where {Nb,Ne,TU,TF}

    @unpack g, vortices, _Rvmat = vortexmodel

    w = TU()
    computew!(w,vortexmodel)

    return w
end

"""
    solvesystem!(sol::PotentialFlowSolution, vortexmodel::VortexModel, wphysical::Nodes; Ub=(0.0,0.0), U∞=(0.0,0.0), Γb=nothing, σ=SuctionParameter.(zeros(Ne)))

Computes the potential flow solution `sol` of the system consisting of the bodies and vortices in `vortexmodel` on the physical grid. The concrete type of the solution `sol` has to agree with the model. If the system has no regularized bodies, `sol` has to be a `UnregularizedPotentialFlowSolution`. If the sytem has regularized bodies, `sol` has to be a `SteadyRegularizedPotentialFlowSolution` or `UnsteadyRegularizedPotentialFlowSolution`.

Translational body motion can be specified with the optional array of tuples `Ub`, which has to contain as many elements as number of bodies in the model.

Rotational body motion can be specified with the optional array `Ωb`, which has to contain as many elements as number of bodies in the model.

Bound circulation can be specified with the optional array `Ωb`, which has to contain as many elements as number of bodies in the model.

A uniform flow can be specified with the optional tuple U∞.

σ
"""
function solvesystem!(sol::PotentialFlowSolution, vortexmodel::VortexModel{Nb,Ne,TU,TF}, wphysical::TU; Ub::Union{Tuple{Float64,Float64},Array{Tuple{Float64,Float64}}}=fill((0.0,0.0), Nb), U∞=(0.0,0.0), Γb=nothing, σ=SuctionParameter.(zeros(Ne))) where {Nb,Ne,TU,TF}

    @unpack g, vortices, bodies, edges, system, _nodedata, _w, _ψb = vortexmodel

    # Assure that Ub is an array
    if Ub isa Tuple
        Ub = [Ub]
    end

    # Because the discrete operators work in index space, we follow the convention in the paper and scale the physical vorticity field wphysical (the approximation to the continuous vorticity field) such that discrete vorticity field _w is is approximately the continuous vorticity multiplied by ∆x.
    _w .= wphysical*cellsize(g)

    # The discrete streamfunction field is constrained to a prescribed streamfunction on the body that describes the body motion. The body presence in the uniform flow is taken into account by subtracting its value from the body motion (i.e. a body motion in the -U∞ direction) and adding the uniform flow at the end of this routine.
    _ψb .= -U∞[1]*(collect(bodies)[2]) .+ U∞[2]*(collect(bodies)[1]);
    for i in 1:Nb
        _ψb[getrange(bodies,i)] .+= Ub[i][1]*(collect(bodies[i])[2]) .- Ub[i][2]*(collect(bodies[i])[1]);
    end
    # Similarly as above, the discrete streamfunction field ψ is approximately equal to the continuous streamfunction divided by ∆x. We therefore divide its continuous constraint by ∆x to get the discrete constraint.
    _ψb ./= cellsize(g)

    if !isregularized(vortexmodel)
        #TODO: figure out scaling for Γb
        if isnothing(Γb) && Nb == 1
            Γb = -sum(_w)
            # println("Circulation about body not specified, using Γb = -sum(w)")
        elseif isnothing(Γb) && Nb > 1
            Γb = zeros(Nb)
            # println("Circulation about bodies not specified, using Γb = $(Γb)")
        elseif !isnothing(Γb) # Scale the provided Γb
            Γb = deepcopy(Γb)./cellsize(g)
        end
        rhs = PotentialFlowRHS(_w,_ψb,Γ=Γb)
    elseif issteady(vortexmodel)
        # Use same scaling for σ as for f
        SP = deepcopy(σ)./cellsize(g)
        rhs = PotentialFlowRHS(_w,_ψb,SP)
    else
        # Use same scaling for σ as for f
        SP = deepcopy(σ)./cellsize(g)
        Γw = sum(_w)#*cellsize(g)
        rhs = PotentialFlowRHS(_w,_ψb,SP,Γw)
    end

    ldiv!(sol,system,rhs)

    # The computed discrete streamfunction field ψ is approximately equal to the continuous streamfunction divided by ∆x. We now scale the discrete field back to the physical grid.
    sol.ψ .*= cellsize(g)
    # Because Δψ + Rf = -w, f also has to be scaled back. The result is f = γ*Δs or f̃ = γ̃
    # TODO: need to change this such that it always return f and not f̃. Then we also have to change computeimpulse
    if !isregularized(vortexmodel)
        sol.f .*= cellsize(g)
    else
        sol.f̃ .*= cellsize(g)
    end

    if !issteady(vortexmodel) && isregularized(vortexmodel)
        sol.δΓ_kvec .*= cellsize(vortexmodel.g)
    end

    # Add the uniform flow to the approximation to the continuous stream function field
    xg,yg = coordinates(_nodedata,g)
    sol.ψ .+= U∞[1]*yg' .- U∞[2]*xg

    return sol
end

function solvesystem!(ψ::TU, f::TF, vortexmodel::VortexModel{Nb,Ne,TU,TF}, w::TU; kwargs...) where {Nb,Ne,TU,TF}

    if !isregularized(vortexmodel)
        sol = PotentialFlowSolution(ψ,f)
    elseif issteady(vortexmodel)
        sol = PotentialFlowSolution(ψ,f,zeros(Nb))
    else
        sol = PotentialFlowSolution(ψ,f,zeros(Nb),zeros(Ne))
    end

    solvesystem!(sol,vortexmodel,w;kwargs...)

    return sol
end

function solvesystem(vortexmodel::VortexModel{Nb,Ne,TU,TF}, w::TU; kwargs...) where {Nb,Ne,TU,TF}

    sol = solvesystem!(TU(),TF(),vortexmodel,w;kwargs...)

    return sol
end

function computeψ(vortexmodel::VortexModel{Nb,Ne,TU,TF}; kwargs...)::TU where {Nb,Ne,TU,TF}

    ψ = TU()

    computew!(vortexmodel._w, vortexmodel)
    solvesystem!(ψ, vortexmodel._bodydata, vortexmodel, vortexmodel._w; kwargs...)

    return ψ
end

# function computeψ(vortexmodel::VortexModel{Nb,Ne,TU,TF}; U∞=(0.0,0.0))::TU  where {Nb,Ne,TU,TF}
#
#     @unpack g, bodies, system, _nodedata, _bodydata = vortexmodel
#
#     xg,yg = coordinates(_nodedata,g)
#     _bodydata .= -U∞[1]*(collect(bodies)[2]) .+ U∞[2]*(collect(bodies)[1]);
#     rhs = PotentialFlowRHS(w,_bodydata)
#     sol = PotentialFlowSolution(_nodedata,_bodydata,zeros(Nb),)
#
#     return sol.ψ
# end

# frame of reference
# w and f are physical quantities
# TODO: clean this up
function computeimpulse(vortexmodel::VortexModel{Nb,Ne,TU,TF}, w::TU, f::TF; Ub=(0.0,0.0), U∞=(0.0,0.0), kwargs...) where {Nb,Ne,TU,TF}

    @unpack g, vortices, bodies, _bodydata = vortexmodel

    # Assure that Ub is an array
    if Ub isa Tuple
        Ub = [Ub]
    end

    xg, yg = coordinates(w,g)
    Δx = cellsize(g)

    impulse = [0.0,0.0]
    # Formula 61 (see formula 6.16 in book)
    impulse = [Δx^2*sum(w.*yg'),Δx^2*sum(-w.*xg)]
    # println(impulse)

    v = VectorData(_bodydata)

    v.u .= 0
    v.v .= 0

    for i in 1:Nb
        v.u[getrange(bodies,i)] .+= Ub[i][1]
        v.v[getrange(bodies,i)] .+= Ub[i][2]
    end

    for i in 1:Nb
        impulse += computeimpulsesurfaceintegral(bodies[i], f[getrange(bodies,i)], v.u[getrange(bodies,i)],v.v[getrange(bodies,i)])
    end



    # surfaceintegral_x = ry'*((f./Δs + nx.*v.v - ny.*v.u).*Δs)
    # surfaceintegral_y = -rx'*((f./Δs + nx.*v.v - ny.*v.u).*Δs)

    # P_x = volumeintegral_x + surfaceintegral_x
    # P_y = volumeintegral_y + surfaceintegral_y

    return impulse[1], impulse[2]
end

function computeimpulse(vortexmodel::VortexModel{Nb,0,TU,TF}; kwargs...) where {Nb,TU,TF}

    computew!(vortexmodel._w,vortexmodel)
    solvesystem!(vortexmodel._nodedata, vortexmodel._bodydata, vortexmodel, vortexmodel._w; kwargs...)

    # We have to recalculate vortexmodel._w because it gets modified in solvesystem
    computew!(vortexmodel._w,vortexmodel)
    P_x, P_y = computeimpulse(vortexmodel, vortexmodel._w, vortexmodel._bodydata; kwargs...)

    return P_x, P_y
end

function computeimpulse(vortexmodel::VortexModel{Nb,Ne,TU,TF}; kwargs...) where {Nb,Ne,TU,TF}

    computew!(vortexmodel._w,vortexmodel)
    solvesystem!(vortexmodel._nodedata, vortexmodel._bodydata, vortexmodel, vortexmodel._w; kwargs...)

    # We have to recalculate vortexmodel._w because it gets modified in solvesystem
    computew!(vortexmodel._w,vortexmodel)

    # vortexmodel._bodydata is f̃, so we have to multiply by f₀
    vortexmodel._bodydata .= vortexmodel._bodydata.*vortexmodel.system.f₀

    P_x, P_y = computeimpulse(vortexmodel, vortexmodel._w, vortexmodel._bodydata; kwargs...)

    return P_x, P_y
end

function computeimpulsesurfaceintegral(body::Body{N,RigidBodyTools.ClosedBody}, f, u, v) where {N}
    rx,ry = collect(body)
    nx,ny = normalmid(body)
    Δs = dlengthmid(body)
    return [ry'*((f./Δs + nx.*v - ny.*u).*Δs), -rx'*((f./Δs + nx.*v - ny.*u).*Δs)]
end

function computeimpulsesurfaceintegral(body::Body{N,RigidBodyTools.OpenBody}, f, u, v) where {N}
    rx,ry = collect(body)
    return [ry'*f, -rx'*f]
end

# function computevortexvelocities(vortexmodel::VortexModel{Nb,Ne,TU,TF},ψ::TU,Emat) where {Nb,Ne,TU,TF}
#
#     @unpack g, vortices = vortexmodel
#
#     q = NodePair(Dual,Dual,size(g))
#     curl!(q,ψ)
#
#     Ẋ = VectorData(length(collect(vortices)[1]));
#
#     Ẋ.u .= Emat*q.u
#     Ẋ.v .= Emat*q.v
#
#     return Ẋ
#
# end

# function curl!(nodepair::NodePair{Dual, Dual, NX, NY},
#                s::Nodes{Dual,NX, NY}) where {NX, NY}
#
#     view(nodepair.u,2:NX-1,2:NY-1) .= 0.5*(view(s,2:NX-1,3:NY) .- view(s,2:NX-1,1:NY-2))
#     #@inbounds for y in 1:NY-1, x in 1:NX
#     #    edges.u[x,y] = s[x,y+1] - s[x,y]
#     #end
#
#     view(nodepair.v,2:NX-1,2:NY-1) .= 0.5*(view(s,1:NX-2,2:NY-1) .- view(s,3:NX,2:NY-1))
#     #@inbounds for y in 1:NY, x in 1:NX-1
#     #    edges.v[x,y] = s[x,y] - s[x+1,y]
#     #end
#     nodepair
# end
