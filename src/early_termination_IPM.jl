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
function early_termination(solver, best_ub,ldltS::Union{SuiteSparse.CHOLMOD.Factor,Nothing},debug_print=false,N=0)
    # check κ/τ before normalization
    ktratio = solver.info.ktratio
    if ktratio <= 1e-2 # if ktratio <= 1e-2, then problem is feasible 
        data = solver.data
        variables = solver.variables
        if ~isnothing(ldltS) #this enables optimisation-based correction
            dual_cost = compute_optimisation_based_dual_cost(data, variables,solver.residuals,solver.info,ldltS,debug_print)
        else
            dual_cost = compute_dual_cost(data, variables,solver.residuals,solver.info,debug_print,N) #compute current dual cost
        end
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
"""
Dual cost computation for early termination
"""

"""Optimsation-based correction """
function optimise_correction(data,ldltS, x_k, residual_k,G,h,Ib)
    v = [-residual_k; -Ib*x_k; h-G*x_k] 
    corr = ldltS \ v
    return corr
end
function compute_optimisation_based_dual_cost(data, variables,residuals,info,ldltS,debug_print=false)
    
    n = data.n # assuming all-integer variables
    nu = 6
    nx = 12
    N = Int((n-2*nu)/(3*nu)) # horizon number e.g. 2,4,6, 8 (we have number of vars = N*nu + (N+1)*nx)
    g_width = (N+1)*nx # Gx=h : the system dynamics over N=1,2 (2*nx) and initial state (another nx)
    ib_width = N*nu
    τinv = inv(variables.τ)
    x = variables.x * τinv # normalize by τ
    y = τinv*variables.z[g_width+1:end] # dual variables for inequality (NonnegativeConeT) constraints
    z = τinv*variables.z[1:g_width] # dual variables for equality (ZeroConeT) constraints

    # decompose data.A and data.b
    #recall: Ã = vcat(G, A, -Ib, Ib)  and b̃ = vcat(h, b, -lb, ub)[:]
    G = data.A[1:g_width,:] 
    A0 = data.A[g_width+1:end-2*ib_width, :] 
    Ib = data.A[end-ib_width+1:end,:] # last N*nu rows of Ã
    h = data.b[1:g_width]
    b0 = data.b[g_width+1:end-2*ib_width]
    neg_l = data.b[end-2*ib_width+1:end-ib_width] #careful of sign! "-l" in paper is the data we extract from b already!
    u = data.b[end-ib_width+1:end]
    y0 = y[1:end-2*ib_width] 
    yminus = y[end-2*ib_width+1:end-ib_width] # last 2m rows of cones are the lower and upper bounds updated at each branching
    yplus = y[end-ib_width+1:end] # yplus corresponds to s_(+) or lower bounds, yminus to s_(-) or upper bounds
    
    # define the linear residual_k as in paper eq(10)
    r_k = Symmetric(data.P)*x + data.q + G'*z + A0'*y0 + Ib'*(yplus - yminus)
     
    Δx = zeros(length(x))
    Δy = zeros(length(yplus))
    Δz = zeros(length(z))
    corr = optimise_correction(data,ldltS,x,r_k,G,h,Ib) # see section III.C
    Δx .= corr[1:length(x)]
    Δy .= corr[length(x)+1:end-length(z)]  
    Δz .= corr[end-length(z)+1:end]

    Δyplus = max.(zeros(length(Δy)),Δy) 
    cor_x = x + Δx 
    cor_yplus = yplus + Δyplus
    cor_yminus = yminus + Δyplus- Δy
    cor_z = z + Δz
    # compute dual cost as given by last dual formulation in Section II.B in Yuwen's paper
    dual_cost = -0.5*cor_x'*Symmetric(data.P)*cor_x - h'*cor_z - b0'*y0 - u'*yplus - neg_l'*yminus + (-neg_l-u)'*(Δyplus) + neg_l'*Δy

    if debug_print
        println("residuals ", norm(residuals.rx*τinv- r_k,Inf))
        con1 = cor_yminus >=zeros(length(yplus)) 
        con2 = cor_yplus >= zeros(length(yplus)) 
        con4 = Δyplus >= zeros(length(yplus)) 
        con5 = (Δyplus- Δy) >= zeros(length(yplus)) 

        if con1 && con2 && con4 && con5
            printstyled("dual constraints satisfied!\n", color = :green)
        else 
            printstyled("error dual constraints ",con1, con2,con4,con5,"\n", color = :red)
            error("Stop")
        end
    end
    diff =  info.cost_dual-dual_cost
    if diff > 1e-6 && debug_print
        printstyled("different dual cost ", dual_cost, " ", info.cost_dual, "\n",color = :red)
    elseif diff < 0 
        printstyled("info.cost_dual is smaller than dual cost ", dual_cost, " ", info.cost_dual,"\n",color = :red)
        
    end
    

    return dual_cost
    
end

"""Basic correction only on dual variables"""
function compute_dual_cost(data, variables,residuals,info,debug_print=false, N=0) 
    # info.cost_dual   =  (-residuals.dot_bz*τinv - xPx_τinvsq_over2)/cscale
    # this is (-data.b'*variables.z*τinv -0.5*xPx*τinv^2) / data.equilibration.c[]
    n = data.n # assuming all-integer variables
    τinv = inv(variables.τ)
    x = variables.x * τinv # normalize by τ
    y = variables.z*τinv #include sign difference with Clarabel where z >= 0 but y_l and y_u are nonpositive
    # correction by yminus and yplus (Method 2, auxiliary optimization)
    # yplus corresponds to s_(+) or lower bounds, yminus to s_(-) or upper bounds
    if ~iszero(N)
        println("Portfolio early termination...........................")
        yplus = y[end-2-N-n+1:end-2-N] # upper bound constraints
        yminus = y[end-2-N-2*n+1:end-2-N-n] # lower bound constraints
        u = data.b[end-2-N-n+1:end-2-N] 
        neg_l = data.b[end-2-N-2*n+1:end-2-N-n]
        A0 = vcat(data.A[1:end-2-N-2*n,:],data.A[end-2-N+1:end,:])
        b0 = vcat(data.b[1:end-2-N-2*n],data.b[end-2-N+1:end])
        y0 = vcat(y[1:end-2-N-2*n],y[end-2-N+1:end])
        
    else
        yminus = y[end-2*n+1:end-n] # last 2m rows of cones are the lower and upper bounds updated at each branching
        yplus = y[end-n+1:end]
        neg_l = data.b[end-2*n+1:end-n] #careful of sign! "-l" in paper is the data we extract from b already!
        u = data.b[end-n+1:end]
        A0 = data.A[1:end-2*n, :] 
        b0 = data.b[1:end-2*n]
        y0 = y[1:end-2*n] 
    end    
    residual_k = Symmetric(data.P)*x + data.q + A0'*y0 + yplus - yminus
    Δx = zeros(length(x))
    Δy = zeros(length(yplus))
        
    #dual correction, only for Δy = Δyplus - Δyminus
    Δy .= -residual_k 
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
    diff =  info.cost_dual-dual_cost
    if diff > 1e-6 && debug_print
        printstyled("different dual cost ", dual_cost, " ", info.cost_dual, "\n",color = :red)
    elseif diff < 0 
        printstyled("info.cost_dual is smaller than dual cost ", dual_cost, " ", info.cost_dual,"\n",color = :red)
        
    end
    

    return dual_cost
end




