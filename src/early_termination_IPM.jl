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
function early_termination(solver, best_ub,η,debug_print=false)
    # check κ/τ before normalization
    ktratio = solver.info.ktratio
    if ktratio <= 1e-2 # if ktratio <= 1e-2, then problem is feasible 
        data = solver.data
        variables = solver.variables
        dual_cost = compute_dual_cost(data, variables,solver.residuals,solver.info,η,debug_print) #compute current dual cost
        if debug_print
            println("Found dual cost: ", dual_cost)
        end
        if (dual_cost > best_ub)
            printstyled("early_termination has found dual_cost larger than best ub \n", color = :red)
            solver.info.status = Clarabel.EARLY_TERMINATION
            return true
        end
    end 
    
    return false
end
function optimise_correction(data, x_k, residual_k,s_k, η)
    v = [-residual_k; -x_k] 
    eyemat=Matrix(1.0I, length(x_k), length(x_k)) 
    ldltS = ldlt([Symmetric(data.P) eyemat ;
            eyemat   -η*eyemat])
    corr = ldltS \ v
    return corr
end
"""
Dual cost computation for early termination
"""
#new framework for dual cost computation, 

function compute_dual_cost(data, variables,residuals,info, η,debug_print=false) 
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
    residual_k = Symmetric(data.P)*x + data.q + A0'*y0 + yplus - yminus
    Δx = zeros(length(x))
    Δy = zeros(length(yplus))
    if η != 1000.0 #this enables optimisation-based correction
        corr = optimise_correction(data,x,residual_k, variables.s*τinv,η) # see section III.C
        Δx .= corr[1:length(x)]
        Δy .= corr[length(x)+1:end]
    else # disabled optimisation based correction
    #dual correction, only for Δy = Δyplus - Δyminus
        Δy .= -residual_k
    end
    # value of residual before the correction

    Δyplus = max.(zeros(length(Δy)),Δy) 
    cor_x = x + Δx 
    cor_yplus = yplus + Δyplus
    cor_yminus = yminus + Δyplus- Δy
    dual_cost = -0.5*cor_x'*Symmetric(data.P)*cor_x - b0'*y0 - u'*yplus - neg_l'*yminus + (-neg_l-u)'*(Δyplus) + neg_l'*Δy

    if debug_print
        println("residuals ", norm(residuals.rx*τinv- residual_k,Inf))
        con1 = cor_yminus >=zeros(length(yplus)) 
        con2 = cor_yplus >= zeros(length(yplus)) 
        con4 = Δyplus >= zeros(length(yplus)) 
        con5 = (Δyplus- Δy) >= zeros(length(yplus)) 

        con3 = isapprox(norm(Symmetric(data.P)*cor_x + data.q + A0'*y0 + cor_yplus - cor_yminus),0.0, atol=1e-6)
        if con1 && con2 && con3 && con4 && con5
            printstyled("dual constraints satisfied!\n", color = :green)
        else 
            printstyled("error dual constraints ",con1, con2, con3,"\n", color = :red)
            error("Stop")
        end
    end
    diff =  norm(dual_cost - info.cost_dual, Inf)
    if diff > 1e-6 && debug_print
        printstyled("different dual cost ", dual_cost, " ", info.cost_dual, "\n",color = :red)
    end
    

    return dual_cost
end




