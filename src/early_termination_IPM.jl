using SparseArrays, LinearAlgebra
T = Float64
#include("direct_Clarabel_large_augmented.jl")
"""
	MIClarabel

IPM solves MIQP: 
min 0.5 x'Px + q'x
s.t.  Ax + s == b,
	  x ∈ Z, relaxed to [lx, ux]
      s ∈ ClarabelCones
rewritten as
s.t.  [A; I; -I]x + [s; s_; s+] == [b; u; -l]
		s ∈ S, 
        s_ and s+ are ClarabelNonnegativeCone

Dual:
max(wrt x, y, y(_+), y_) : -0.5 x'Px - b'*y - y(_+)'*u + y_'*l 
s.t.	Px + q + A'y + y_u - y_l == 0
		y ∈ C*
		y_u and y_l >= 0
"""

"""early termination
- best_ub: current best upper bound on objective value stored in root node
- node: current node, needs to be implemented with Clarabel solver
"""
function early_termination(solver, best_ub,debug_print=false)
    # check κ/τ before normalization
    ktratio = solver.info.ktratio
    if ktratio <= 1e-2 # if ktratio <= 1e-2, then problem is feasible 
        data = solver.data
        variables = solver.variables
        dual_cost = compute_dual_cost(data, variables,solver.residuals,solver.info,debug_print) #compute current dual cost
        if debug_print
            println("Found dual cost: ", dual_cost)
        end
        if (dual_cost > best_ub)
            printstyled("early_termination has found dual_cost larger than best ub \n", color = :red)
            solver.info.status = Clarabel.EARLY_TERMINATION
            # TOASK or: solver.solution.status =  Clarabel.EARLY_TERMINATION
            return true
        end
    end 
    
    return false
end

"""
Dual cost computation for early termination
"""
#new framework for dual cost computation, 
# We can use qdldl.jl for optimization (17) later on.

function compute_dual_cost(data, variables,residuals,info, debug_print=false) 
    # info.cost_dual   =  (-residuals.dot_bz*τinv - xPx_τinvsq_over2)/cscale
    # this is (-data.b'*variables.z*τinv -0.5*xPx*τinv^2) / data.equilibration.c[]

    m = data.n # assuming all-integer variables
    τinv = inv(variables.τ)
    x = variables.x * τinv # normalize by τ
    y = variables.z*τinv #include sign difference with Clarabel where z >= 0 but y_l and y_u are nonpositive
    # correction by yminus and yplus (Method 2, auxiliary optimization)
    # yplus corresponds to s_(+) or lower bounds, yminus to s_(-) or upper bounds
    yminus = y[end-2*m+1:end-m] # last 2m rows of cones are the lower and upper bounds updated at each branching
    yplus = y[end-m+1:end]
    neg_l = data.b[end-2*m+1:end-m] #careful of sign! "-l" in paper is the data we extract from b already!
    u = data.b[end-m+1:end]
    A0 = data.A[1:end-2*m, :] 
    b0 = data.b[1:end-2*m]
    y0 = y[1:end-2*m] 

    Δx = zeros(length(x))

    # value of residual before the correction
    residual = Symmetric(data.P)*x + data.q + A0'*y0 + yplus - yminus

    #dual correction, only for Δy = Δyplus - Δyminus
    Δy = -residual 
    Δyplus = max(zeros(length(Δy)),Δy) 
    #mul!(Δx, op.P, coef*Δy) 
    cor_x = x + Δx # for simplicity, no correction for x

    if debug_print
        println("residuals ", norm(residuals.rx*τinv- residual,Inf))
    end
    dual_cost = -0.5*cor_x'*Symmetric(data.P)*cor_x - b0'*y0 - u'*yplus - neg_l'*yminus + (-neg_l-u)'*(Δyplus) + neg_l'*Δy
    diff =  norm(dual_cost - info.cost_dual, Inf)
    if diff > 1e-6 && debug_print
        printstyled("different dual cost ", dual_cost, " ", info.cost_dual, "\n",color = :red)
    end
    

    return dual_cost
end


"""
Penalized correction (ADMM paper)

min 0.5*α*||Δx||^2 + 0.5*γ*||Δy||^2
s.t.    P*Δx - Δy = -residual # NOTE: this is just my guess ?
where   residual := Px - A'*y - y + q
"""
function dual_correction!(data, Δx, Δy, residual)
    n = length(data.q)
    #coef = γ/α
    coef = 100
    #solve correction matrix, correction on y^k and x^k
    mat = coef*op.P*op.P + I
    IterativeSolvers.cg!(Δy, mat, -residual) # this solves (P^2*coef+I)*Δy = -r (eq.18)
    mul!(Δx, op.P, coef*Δy)
end


