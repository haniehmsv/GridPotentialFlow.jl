export pressurejump!, pressure!, velocity!, surface_velocity!, sided_pressures

function pressurejump!(dp::ScalarData{N},γn::ScalarData{N},γnp1::ScalarData{N},v̄s::VectorData{N},Δt::Real,sys::ImmersedLayers.ILMSystem{static,<:GridPotentialILMProblem}) where {N,static}
    @unpack base_cache, extra_cache = sys
    @unpack g, nrm, gcurl_cache, sdata_cache = base_cache
    @unpack CLinvCT, Rn = extra_cache
    gcurl_cache .= Rn*γn
    gcurl_cache .= cellsize(g)*(Rn*γnp1 - gcurl_cache)/Δt
    inverse_laplacian!(gcurl_cache,sys)
    surface_curl!(sdata_cache,gcurl_cache,sys)
    dp .= CLinvCT\sdata_cache
    dp .*= -1.0

    cross!(sdata_cache,nrm,v̄s)
    # dp .-= sdata_cache∘γn
    return dp, -sdata_cache∘γn

end

"""An alternative way to calculate pressure jump across the plate using the idea of Δϕₖ=-Γₖ"""

function pressurejump!(dp::ScalarData{N},γn::ScalarData{N},vm1::VortexModel,γnp1::ScalarData{N},v̄s::VectorData{N},Δt::Real,Δs, sys::ImmersedLayers.ILMSystem{static,<:GridPotentialILMProblem}) where {N,static}
    @unpack base_cache, extra_cache = sys
    @unpack g, nrm, gcurl_cache, sdata_cache = base_cache
    @unpack CLinvCT, Rn = extra_cache

    for i in 1:length(dp)
        dp[i] = sum(view(γnp1,1:i).*Δs[1:i]) + vm1.vortices.Γ[end-1]
        dp[i] -= sum(view(γn,1:i).*Δs[1:i])
    end
    dp ./= -Δt
    cross!(sdata_cache,nrm,v̄s)
    dp .-= sdata_cache∘γn
end

function velocity!(v̄::Edges{Primal,NX,NY},ψ::Nodes{Dual,NX,NY},sys::ImmersedLayers.ILMSystem) where {NX,NY}
    @unpack base_cache = sys
    @unpack g = base_cache
    curl!(v̄,ψ)
    v̄ ./= cellsize(g)
end

function surface_velocity!(v̄s::VectorData,v̄::Edges{Primal},sys::ImmersedLayers.ILMSystem)
    @unpack base_cache = sys
    @unpack Esn = base_cache
    v̄s .= Esn*v̄
end

function pressure!(p̄::Nodes{Primal,NX,NY},v̄::Edges{Primal,NX,NY},dp::ScalarData,sys::ImmersedLayers.ILMSystem) where {NX,NY}
    @unpack base_cache, extra_cache = sys
    @unpack g, L, Rsn, nrm, gsnorm_cache, snorm_cache = base_cache
    @unpack v_cache, cv_cache = extra_cache

    fill!(v_cache,0.0)
    ImmersedLayers.convective_derivative!(v_cache,v̄,base_cache,cv_cache)
    product!(snorm_cache,nrm,dp)
    gsnorm_cache .= Rsn*snorm_cache
    v_cache .-= gsnorm_cache

    fill!(p̄,0.0)
    divergence!(p̄,v_cache)
    p̄ ./= cellsize(g)
    inverse_laplacian!(p̄,sys)
    p̄ .*= -1.0
end

function sided_pressures(p̄::Nodes{Primal},dp::ScalarData,sys::ImmersedLayers.ILMSystem)
    @unpack base_cache,extra_cache = sys
    @unpack sdata_cache = base_cache
    @unpack Ec = extra_cache

    sdata_cache .= Ec*p̄
    return sdata_cache + 0.5*dp, sdata_cache - 0.5*dp
end
"""
    cross!(C::VectorData,A::ScalarData/VectorData,B::VectorData/ScalarData) -> VectorData

Compute the cross product between the point data `A` and `B`, one of which
is scalar data and treated as an out-of-plane component of a vector, while
the other is in-plane vector data, and return the result as vector data `C`.
"""
function cross!(C::VectorData{N},A::ScalarData{N},B::VectorData{N}) where {N}
    @. C.u = -A*B.v
    @. C.v = A*B.u
    return C
end

function cross!(C::ScalarData{N},A::VectorData{N},B::VectorData{N}) where {N}
    @. C = A.u*B.v-A.v*B.u
    return C
end

function cross!(C::VectorData{N},A::VectorData{N},B::ScalarData{N}) where {N}
    @. C.u = A.v*B
    @. C.v = -A.u*B
    return C
end