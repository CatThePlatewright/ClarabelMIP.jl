import QDLDL

struct QDLDLDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}

    factors::QDLDL.QDLDLFactorisation{T, Int}

    function QDLDLDirectLDLSolver{T}(KKT::SparseMatrixCSC{T},Dsigns,settings) where {T}

        dim = LinearAlgebra.checksquare(KKT)

        #make a logical factorization to fix memory allocations
        factors = QDLDL.qdldl(
            KKT;
            Dsigns = Dsigns,
            regularize_eps   = settings.dynamic_regularization_eps,
            regularize_delta = settings.dynamic_regularization_delta,
            logical          = true
        )

        return new(factors)
    end

end

DirectLDLSolversDict[:qdldl] = QDLDLDirectLDLSolver
required_matrix_shape(::Type{QDLDLDirectLDLSolver}) = :triu

#update entries in the KKT matrix using the
#given index into its CSC representation
function update_values!(
    ldlsolver::QDLDLDirectLDLSolver{T},
    index::AbstractVector{Int},
    values::Vector{T}
) where{T}

    #Update values that are stored within
    #the reordered copy held internally by QDLDL.
    QDLDL.update_values!(ldlsolver.factors,index,values)

end

#scale entries in the KKT matrix using the
#given index into its CSC representation
function scale_values!(
    ldlsolver::QDLDLDirectLDLSolver{T},
    index::AbstractVector{Int},
    scale::T
) where{T}

    QDLDL.scale_values!(ldlsolver.factors,index,scale)

end


#refactor the linear system
function refactor!(ldlsolver::QDLDLDirectLDLSolver{T}, K::SparseMatrixCSC) where{T}

    # K is not used because QDLDL maintains
    # the update matrix entries for itself using the
    # offset/update methods implemented above.
    QDLDL.refactor!(ldlsolver.factors)

    return all(isfinite, ldlsolver.factors.Dinv.diag)

end


#solve the linear system
function solve!(
    ldlsolver::QDLDLDirectLDLSolver{T},
    x::Vector{T},
    b::Vector{T}
) where{T}

    #solve in place 
    @. x = b
    QDLDL.solve!(ldlsolver.factors,x)

end
