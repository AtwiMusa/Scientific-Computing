### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 9e67d263-3662-4fb0-9054-2afd3d38b851
begin 
	using Printf
	using Pkg
	Pkg.activate(mktempdir())
	Pkg.add(["Plots","PyPlot","PlutoUI","ExtendableGrids","GridVisualize", 			       "VoronoiFVM","Triangulate","SimplexGridFactory","DataFrames", "HypertextLiteral", "PlutoVista"])
	using Plots,PlutoUI,PyPlot,ExtendableGrids,VoronoiFVM,GridVisualize,Triangulate,SimplexGridFactory,DataFrames,HypertextLiteral, PlutoVista
	PyPlot.svg(true)
	using LinearAlgebra
end

# ╔═╡ 215aa828-34aa-42f9-a0e0-0e7a46c81cd9
Pkg.add("DifferentialEquations");using DifferentialEquations

# ╔═╡ c91b6729-809c-47da-b47e-0bf41808230b
using Statistics

# ╔═╡ 07f7fd76-c0eb-42b1-843e-0ff67138070a
using InteractiveUtils

# ╔═╡ 281bdc34-664f-4aaf-83ec-53e89cfdb4d8
md"
******
"

# ╔═╡ e112e366-9463-464d-b878-699475ee48a5
md" 
# Gas Transport In Porous Medium
**Authors**: Moussa Atwi and Surya Narayanan Iyer

***

**Supervisor**: Dr. Jürgen Fuhrmann

***

**Group 10**: Master in Scientific Computing

***

**ID**: 464241  and 464252


"

# ╔═╡ 4d88b926-9543-11ea-293a-1379b1b5ae64
md"# Introduction
!!! tip \"Mathematical Background\"
The ``\textit{Porous Medium Equation}`` is a nonlinear degenerate parabolic PDE of second order

$$u_t - \Delta(u^m)  = 0, \quad m > 1. \qquad\qquad (1)$$ 

For signed solutions our diffusion equation should be written as 

$$u_t - \Delta(|u|^{m-1} u) = 0, \quad m > 1$$
- Equation ``(1)`` degenerates for $u = 0$
- Assuming $u \geq 0, (1)$ is formally parabolic only at points where $u > 0$

**In particular**:

1) For $m = 1$ we have the classical heat equation 

$$u_t = \Delta u$$

2) For $m = 2$ we have Boussinesq’s equation (groundwater infiltration)

$$u_t = \Delta u^2$$ 
"

# ╔═╡ c46006d6-085d-487c-a7cd-91b8126fb960
#It is natural to consider the nonnegative solutions from the viewpoint of Physics
# and its formally parabolic only at the non vanishing pts 
# slow diffusion 
#as we are dealing with concen ....

# ╔═╡ 76a630d5-f675-4e51-9b7a-ee357264e9f6
md" 
!!! note \"Barenblatt Solution\"
Equation $(1)$ has a radially symmetric exact solution in $\mathbb{R^d} × (0, \infty),$ the so-called $Barenblatt$ solution


$$b(x,t)= \max\left(0,t^{-\alpha}\left(1-\frac{\beta(m-1)r^{2}}{4m\,t^{\beta}}\right)^{\frac{1}{m-1}}\right), \qquad (2)$$

Where, 

$$r=|x|, \quad \beta =\frac{2\alpha}{d} \hspace{0.3cm}\text{and}\hspace{0.3cm}\alpha=\frac{1}{m-1+\frac{2}{d}}.$$

Barenblatt solution spreads a finite amount of mass over the space domain, thus for any fixed $t \in (0, T),\;b(x, t)$ has a compact support 

$$|x|^2 \leq \frac{4m\,t^{\beta}}{\beta (m - 1)}$$ which grows as $t$ grows.

"


# ╔═╡ 3c3a3d5f-17c2-4f37-8b3e-4ded02650431
#The effect of the nonlinearity in $(1)$ is that there are solutions with compact support which is the case for a bareblatt solutiuon with a compact support x^2 ....

#get wider as m increase 

# ╔═╡ 125329a8-530c-41ff-83c2-ea27e29891e7
md"
!!! tip \"Physical Background\"

The nonlinear diffusion equation can be written in a more generalized form as 

$$u_t - \nabla\cdot(D(u)\nabla u) = f(x, t) \iff  u_t + \nabla\cdot\vec{j} = f(x, t)$$

where

- ``D(u)=mu^{m-1}:`` Density-dependent diffusivity.

- ``\vec j = - D(u)\nabla u = - mu^{m-1}\nabla u:`` The vector field of the diffusion species flux 
"

# ╔═╡ 9d89499f-ea4d-4548-9b30-e0f8f0a02ca9
# We may write the nonlinear diffusion equation as a heat equation  also with flux expression and in a more generailzed form 
#- ``f(x, t):`` Species sources
#- ``u(x, t):`` The time depend local amount of (species concentration, temp, density)

#- ``D(u)=mu^{m-1}:`` Density-dependent diffusivity.(diffusion coefficient depends on the solution )

#- ``\vec j = - D(u)\nabla u = - mu^{m-1}\nabla u:`` The (vector field) of the diffusion (species flux) 

# ╔═╡ d47e0e73-8909-4dad-b9e7-380c63033298
md"""
!!! note \"Where might porous medium equation arise\"

The $porous\; medium\; equation$ owes its name to its use in describing the flow of
gas in a porous medium. The flow is governed by the following laws:\


1) Conservation of mass: 

$$\epsilon\rho_t + \nabla\cdot(\rho \vec{V}) = 0$$ 

2) Darcy's law:

$$\vec{V} = -\frac{k}{\mu}\nabla P$$


3) Equation of state, which for perfect gases asserts that:

$$\rho = \rho_0 P^{\gamma}$$

Eliminating $\vec{V}$ and $P$ gives:

$$\rho_t = \Delta \rho^{1 + \frac{1}{\gamma}}$$ where $m = 1 + \frac{1}{\gamma} \geq 2$ 
"""

# ╔═╡ 96c4b528-273a-43e0-8256-9e2ab0bb59ef
#1) Conservation of mass: 

# $\epsilon\in [0 , 1]$ is the porosity of the medium, $\rho$ density of gas and $\vec{V}$ velocity of fluid. 

#If $\epsilon = 1$, we have the classical conservation of mass, 
#that is the flow is every where.

#2) Darcy's law:

#which means that the gradient of the pressure causes the gas to flow (the flow ofa gas ). 

#Here $P$ is the pressure, $k$ is the permeability of the medium, $\mu$ is the viscosity of the fluid and 

#the minus sign means that the flow goes from low to high pressure.

#3) Equation of state, which for perfect gases asserts that:

#where $\gamma$, is the so-called $polytropic\;exponent$ with $0 < \gamma \leq 1$. 

#and $\rho_0$ is the reference pressure. 
# for gamma = 1  we have a linear relation with the pressure
#Eliminating $\vec{V}$ and $P$ gives:


#Setting $u = \rho$ and rescaling $t$ by new time $t' = Ct\;$ we get

#$$u_t = \Delta u^m, \quad m> 1.$$ 

# ╔═╡ ed2910e2-9f07-4024-af44-e360a7f831db
md"""
For our problem, we choose m=$(@bind m Scrubbable(1:1:10,default=2)).
"""

# ╔═╡ 96bae2ce-cc36-414d-8bcd-5ddd4b88a06a
md"""
!!! tip \"Time dependent Robin BVP\"
Consider a polygonal bounded domain ``\Omega \subset \mathbb{R^d}``. Choose final time ``T > 0.`` Regard functions ``u(x, t) \rightarrow \mathbb{R}`` with ``u(x, 0) = u_0(x)``.

```math
\begin{aligned}
u_t - \nabla\cdot (D\nabla u) &=0 \quad \text{in} \;\;\Omega \times[0,T] \\
    D \nabla u \cdot \vec{n} + \alpha u &= \beta\quad \text{on} \;\;\Gamma\times[0,T]
  \end{aligned}
``` 

where, ``\alpha, \beta \geq 0.`` 

"""

# ╔═╡ ceb9f9e2-5040-477b-a952-78c1015832a5
#for solvability we need additional conditions: initial cond and bdry cond

# ╔═╡ 6a24a9a6-c522-4489-9cc8-75fe3f23a5b0
md"# Implementing the Numerical Methods
To solve our problem:

- We need finite dimensional approximation. We will choose **voronoi FVM**.

- First we have to define a discretization.

- Discretize on time and space. We will use the **Rothe method**. 

***
"

# ╔═╡ 8aac817b-51e5-4b6a-b0f3-4a4c0e4ee42a
#Essentially, we have an infinite number of unknowns (an unknown value in each point of domain), which is not possible to handle on the computer, because of finite computational capabilities and memory. 

#Therefore, we need a finite-dimensional approximation for this PDE. 

#for this purpose We will use the **Voronoi finite volume method** .


#The usual way of defining such an approximation is based on subdividing our computational domain into a finite number of elementary closed subsets (control volumes), which are called \"meshes\" or \"grids\". In our case, these elementary shapes are simplexes (triangles or tetrahedra).

#Moreover the partial differential equation depends on time, we will have to discretize in time and in space. To do that, we will use the Rothe method, which consists to first discretize in time and then in space.

# ╔═╡ 78c42926-8a64-4244-8f4e-5d6aad7dc19d
# first we have to generate the mesh




# ╔═╡ 9392a40c-2a12-438a-906e-2ac036599259
md"## Finite volume space discretization
!!! note \"Mesh Generation\"
This would be done by the package **ExtendableGrids.jl**. 
"

# ╔═╡ ebe0baeb-6ebd-4573-91cd-281d3b0de9d7
md"### 1D Discretization Grid
1D grids are created from vectors of monotonically increasing coordinates using the **simplexgrid** method. Grid in domain $\Omega = (-1, 1),$ consisting of N11=$(@bind N11 Scrubbable(10:5:50,default=31)) points.
"

# ╔═╡ b914b3da-d35e-49d9-aa02-6468508069f5
begin
	
		X17=collect(range(-1,1,length=N11))
		grid1d=simplexgrid(X17)
		gridplot(grid1d,resolution=(500,150),legend=:rt, Plotter = PlutoVista)
	
end

# ╔═╡ d521b91b-d2f4-4737-9c32-b8f6b7a0b05d
#Then for the 2D case, we just need to replace 1D with a 2D tensor product grids 
#This results in the creation of a grid of quadrilaterals. Then, each of them is subdivided into two triangles, resulting in a boundary conforming Delaunay grid. We obtain :

# ╔═╡ 66a6e161-6213-4d91-8750-9a915c827e3a
md"### 2D Discretization Grid
For 2D case, we just need to replace the 1D grid with a 2D grid.\

The grid is created in the domain $\Omega = (-1,1)\times(-1,1)$, consisting of N13=$(@bind N13 Scrubbable(5:5:50,default=15)) points in each coordinate direction, and is a Delaunay-comforming triangulation. 
"

# ╔═╡ 4fee3262-38db-4c23-bf6b-4eddf96c1f02
begin
	um1 = collect(range(-1,1,length=N13))
	grid2d = simplexgrid(um1, um1)
	gridplot(grid2d,size=(500,500), Plotter = PyPlot)
end

# ╔═╡ 668cd97f-1471-4007-adb4-c02bd4b910db
#Once we subdivided the computational domain into a finite number of edges (in 1D) or triangles (in 2D), we can build the Voronoi cells which define the representative elementary volumes (corcontrol volumes) for our finite volume method. In the Delaunay triangulation the voronoi cells respect particular properties which will be useful to solve our problem (e.g. orthogonality of the interfaces between two collocation points, which makes the method consistent).

# ╔═╡ 58049525-8458-4575-93aa-6ae35e26f49e
md"
!!! tip \"Discretization Of Second Order PDE\"
Search function $u: \Omega \times [0, T] \rightarrow \mathbb{R}$ such that $u(x, 0)= u_0(x)$ and 

```math
\begin{aligned}
u_t + \nabla\cdot\vec{j} &= 0\quad \text{in} \;\;\Omega \times[0,T] \\
\vec{j}\cdot\vec n + \alpha u &= \beta\quad \text{on} \;\;\Gamma\times[0,T]
  \end{aligned}
```

where, $\alpha, \beta \geq 0.$



" 

# ╔═╡ f0a06333-19b0-4885-80c8-1687cdc83fb0
#Regard second order PDE with Robin boundary conditions as a system of two first order equations in a Lipschitz domain $\Omega.$

#The computational domain subdivided into finite numberof REV's, thus we can start now discretizing our system and we start by discretizing the continuity equation.

# ╔═╡ b356d4c8-5db1-4ca0-8994-7e31c90b3d68
md"""
!!! note \"Discretization Of Continuity Equation\"
Integrate the spatial derivative in the continuity equation over control volume $\omega{_k}$ and using **Gauss' law**, we get

$\begin{align*}
0 &= u_t + \int_{\omega{_k}}\nabla\cdot\overrightarrow{j}d\omega \\
&= u_t + \int_{\partial{\omega_k}}\vec{j}\cdot\vec{n}{_\omega}\;ds\\
&= u_t+ \sum_{l \in \mathcal{N{_k}}}\int_{\sigma{_{kl}}}\overrightarrow{j}\cdot\vec{n}{_{kl}} \;ds \; + \; \sum_{m\in \mathcal{G{_k}}} \int_{\gamma{_{km}}} \overrightarrow{j}\cdot\overrightarrow{n_m}\;ds\\
\end{align*}$

``\mathcal{G_k}:`` set of non-empty boundary parts of ``\partial\omega_k\;``and``\;\mathcal{N}_{k}:`` set of neighbours of ``\omega_{k}.``
"""

# ╔═╡ efa8ac51-9ab6-4330-b08f-065466b86f08
#Gauss' Divergence theorem is to convert the volume integral over the divergence into a surface integral across the boundaries

#So instead of integrate the fluxes inside the cells, we integrate these fluxes across the boundary of the cells

# ╔═╡ 3c023b94-7386-43f5-8234-9ac20a23826c
md"""
!!! note \"Approximation of flux between control volumes\"
The finite difference approximation of normal derivative is 

$$\nabla u\cdot\vec{n}_{kl} \approx \frac{u_l - u_k}{h_{kl}},$$ then we can approximate the flux between neighboring control volumes

$$\int_{\sigma{_{kl}}}\overrightarrow{j}\cdot\vec{n}{_{kl}} \;ds = - \int_{\sigma{_{kl}}} D(u)\nabla u\cdot\vec{n}{_{kl}}\,ds \approx \frac{|\sigma_{kl}|}{h_{kl}}D(u_k - u_l)$$ Therefore, we get

$\begin{align*}
\sum_{l \in \mathcal{N}_k}{\frac{\sigma{_{kl}}}{h_{kl}}}D(u_k - u_l) + \sum_{m\in \mathcal{G{_k}}} \int_{\gamma{_{km}}} \overrightarrow{j}\cdot\overrightarrow{n_m}\;ds&\approx 0,\\
\end{align*}$

"""

# ╔═╡ f625420f-fb2c-4ee2-86a3-05ed3072cf42
#approximate the fluxes by finite difference using the neigboring control volumes

# ╔═╡ 49cbc1f3-0141-4692-b347-41817322982f
md"""
!!! note \"Approximation of boundary fluxes\"
Assume that $\alpha|_{\Gamma_m} = \alpha_m$ and $\beta|_{\Gamma_m} = \beta_m,$ then the approximation of $\vec{j}\cdot\vec{n}_m$ at the boundary of $\omega_k$ is 

$$\vec{j}\cdot\vec{n}_m \approx \alpha_mu_k - \beta_m$$ and the approximation of the flux from $\omega_k$ through $\Gamma_m$ is 

$$\int_{\gamma{_{km}}} \overrightarrow{j}\cdot\overrightarrow{n_m}\;ds \approx |\gamma_{km}|(\alpha_m u_k - \beta_m)$$

"""

# ╔═╡ 5ba7370a-3bed-4e85-99dc-f26cfe5e280c
md"""
!!! note \"Discretized system of equations\"
The discrete system of equations then writes for $k \in \mathcal{N}:$

$$u_t + \sum_{l \in \mathcal{N}_k}\frac{|\sigma_{kl}|}{h_{kl}}D(u_k - u_l) + \sum_{m\in \mathcal{G{_k}}}|\gamma_{km}|(\alpha_m u_k - \beta_m) = 0,$$ which is equivalent to 

$$u_t + u_k\left( D\sum_{l \in \mathcal{N}_k}\frac{|\sigma_{kl}|}{h_{kl}} +\alpha_m  \sum_{m\in \mathcal{G{_k}}}|\gamma_{km}|\right) - D\sum_{l \in \mathcal{N}_k}\frac{|\sigma_{kl}|}{h_{kl}} u_l - \sum_{m\in \mathcal{G{_k}}}|\gamma_{km}| \beta_m = 0$$


"""

# ╔═╡ 1873c8cf-cdf0-4281-955f-43fe6de92d3f
md"""
## Time Discretization approach
- Choose time discretization points: $0 = t^0 < t^1 < .... < t^N = T.\;$ 
- Let $\tau{^n} = t^n - t^{n-1}$  
- Approximate the time derivative by a finite difference in time.
For $n = 1, 2, 3 .... , N$, solve:

$\frac{u^n - u^{n-1}}{\tau{^n}} - \nabla\cdot D\nabla{u^{\theta}} = 0 \; \text{in} \; {\Omega{\times}[0, T]}$

$D\nabla u^{\theta}\cdot\vec{n} + \alpha u^{\theta}= \beta \; \text{on} \;\partial\Omega{\times}{[0, T]}$ 
where

$$u^{\theta} = \theta{u^n} + (1- \theta){u^{n-1}}$$ 

"""

# ╔═╡ 5444d294-ffb6-4e07-a9b5-c7ee6fae2d19
#$n$ indicates the number of a time step and $\tau^n$ the difference between two neighboring time discretization points or the time step size (length of the present time step)

#Here we use finite differences and more specifically we introduce the so-called One-Step $\theta - \text{scheme}$ for solving time discretization.

# so we replace the time derivative by FD in time and We choose $\theta$ and substitute $u^{\theta}$ in all the terms concerning boundary conditions where. u. theta is. the linear comb of the sol at new and old time pts

#the most important properties for practical computations of this method theta scheme are its strong stability and high-order accuracy.

# ╔═╡ 09ea7716-0027-4974-bff6-8e86e3918365
# we do a 1st time discretization of the problem in a sense of Rothe Method, replace time depnedent derivative by FD in time and intergtare over the control volume

# ╔═╡ c87d6b95-7a60-4a10-a009-f4421a3d1d65
md"""

!!! note \"Derivation of FVM \"

Given control volume $\omega_k,$ integrate equation over space-time control volume $\omega_k \times (t^{n-1} - t^n),\;$ divide by $\tau^n:$

$\begin{align}
0 & = \int_{\omega_k} \left(\frac{1}{\tau^n} (u^n-u^{n-1}) -\nabla \cdot  D \overrightarrow{\nabla} u^{\theta} \right)d\omega \\

 & = \frac{1}{\tau^n}\int_{\omega_k} \left(u^n-u^{n-1}\right) d\omega+ \sum_{l \in \mathcal{N}_k}\frac{|\sigma_{kl}|}{h_{kl}}D(u_k^{\theta} - u_l^{\theta})\\
& + \sum_{m\in \mathcal{G{_k}}}|\gamma_{km}|(\alpha_m u_k^{\theta} - \beta_m)\\
& \approx \frac{|\omega_k|}{\tau^n} ( u^n_k-u^{n-1}_k) +\sum_{l\in N_k} \frac{|\sigma_{kl}|}{h_{kl}} D(u_k^{\theta} - u_l^{\theta})\\
& + \sum_{m\in \mathcal{G{_k}}}|\gamma_{km}|(\alpha_m u_k^{\theta} - \beta_m)
\end{align}$

Then, we obtain 

$$\underbrace{\frac{|\omega_k|}{\tau^n} ( u^n_k-u^{n-1}_k)}_{\rightarrow M} +\underbrace{\sum_{l\in N_k} \frac{|\sigma_{kl}|}{h_{kl}} D(u_k^{\theta} - u_l^{\theta}) + \sum_{m\in \mathcal{G{_k}}}|\gamma_{km}|\alpha_m u_k^{\theta}}_{\rightarrow A}= \underbrace{\sum_{m\in \mathcal{G{_k}}}|\gamma_{km}|\beta_m}_{\rightarrow b}$$


- Resulting matrix equation

$$\frac{1}{\tau^n}\left(Mu^n - Mu^{n - 1} \right) + Au^{\theta} = b$$ 

"""

# ╔═╡ 375770f1-f755-4970-bd5d-f273a2a84bd4
#this results in a matrix equation that looks like that

# ╔═╡ 21d9613f-91d7-4570-aa8c-cfb8871bf8e3
md"""
!!! note \"Nonlinear system of equations\"
We are left with a nonlinear system of equations that must be solved at each time step:

$$u^n + \tau^n M^{-1}\theta A u^n = u^{n-1} + \tau^n M^{-1}(\theta - 1) A u^{n -1} + b\tau^n M^{-1}$$

- ``M=(m_{kl})``, ``A=(a_{kl})`` and ``b=(b_k)`` with coefficients
```math
\begin{aligned}
   a_{kl} &=
   \begin{cases}
     \sum_{l'\in \mathcal N_k} D\frac{|\sigma_{kl'}|}{h_{kl'}} + \sum_{m\in \mathcal{G{_k}}}|\gamma_{km}|\alpha_m& l=k\\
     -D\frac{\sigma_{kl}}{h_{kl}}, & l\in \mathcal N_k\\
     0, & else
   \end{cases}\\ \\
    m_{kl}&= 
   \begin{cases}
      |\omega_k| & l=k\\
      0, & else
   \end{cases}\\ \\

	b_k&= \sum_{m\in \mathcal{G{_k}}}|\gamma_{km}|\beta_m\\
\end{aligned}
```

Turning PDE into nonlinear system which we try to solve.
"""

# ╔═╡ eebd876d-ddd6-4570-a100-d497456abd52
md"""

!!! note \"Matrix Properties\"

To figure out how we solve the nonlinear system we look for matrix properties.

- ``\theta A + M`` is ``strictly\;diagonally\;dominant`` 
$$\sum_{k \neq l} |(\theta A + M)_{kl}| < |(\theta A + M)_{kk}|$$


- ``\theta A + M`` has M $-property$ and it is essentially **invertible**.
"""

# ╔═╡ 5add94c0-41a8-4939-ac2a-97df64a257be
#(off-diagonal entries are non-positive and all of its eigenvalues have positive parts) 

#``A`` and ``M`` have M $-property$ and as the matrix is symmetry then it is essentially **invertible**.

# ╔═╡ 19b8c750-f80e-4b4a-abea-3a6181bad356
md"""
!!! tip \"Lemma\" 
Assume that $A$ has positive main diagonal entries, non-positive off-diagonal entries and row sum zero. Then,  $\|(I + A)^{-1}\|_{\infty} \leq 1$.\


**Note**: In our case: Robin BVP, the matrix does not satisfy row sum zero.
"""

# ╔═╡ 065c038c-b33f-4636-ab12-39ec4a5e4096
#downgrade our problem a little bit to Homogeneous neumann problem. just for the next discussion


# ╔═╡ f6ad658a-4d49-479c-b63d-6f2849c1d926
md"# Transient Neumann BVP

Downgrade our problem a little bit. For $\alpha, \beta = 0$ we get the homogeneous neumann problem:
```math
\begin{aligned}
\partial_t u     -\nabla\cdot D\nabla u &=0 \quad \text{in} \;\;\Omega \times[0,T] \\
    D \nabla u \cdot \vec{n} &= 0\quad \text{on} \;\;\Gamma\times[0,T]
  \end{aligned}
```   

``A`` has row sum zero.

$$\sum_{l=1}^n a_{kl}=0$$

"

# ╔═╡ fd3d5770-4835-4501-b727-3acdcfbab623
md"""
## Physical Properties
!!! note \"Stability Analysis: Implicit vs Explicit\"

Under which conditions we can assume the estimation of new time value by the old time value in the sense of $L^{\infty}$ norm ?

$$\|u^n\|_{\infty} < \|u^{n - 1}\|$$

Regard the matrix equation again:
```math
  \begin{aligned}
    u^n  +  \tau^nM^{-1}\theta Au^n &= u^{n-1}+\tau^nM^{-1}(\theta-1)A u^{n-1}\\
u^n&= (I+\tau^nM^{-1}\theta A)^{-1} (I + \tau^n M^{-1}(\theta - 1)A) u^{n-1}
\end{aligned}
```
From the lemma we have 

$$||(I+\tau^nM^{-1}\theta A)^{-1}||_\infty\leq 1$$ that yields 

$$||u^n||_{\infty} \leq ||B^n u^{n-1}||_\infty$$
thus, we have to estimate $||B||_{\infty}$ where the sufficient condition is that for some $C>0,$

$$(1-\theta)\tau^n \leq Ch^2$$

**Methods Stability**\

-  Implicit Euler: ``\theta=1`` ``\Rightarrow`` unconditional stability 
-  Explicit Euler: ``\theta=0`` ``\Rightarrow``  ``\tau\leq Ch^2``
-  Crank-Nicolson:  ``\theta=\frac12`` ``\Rightarrow``  ``\tau\leq 2 Ch^2``

"""

# ╔═╡ 04546bf2-f323-4cf8-ad9b-7b3bc72093ce
#for the implicit euler method we immediatly get the stability estimate ||B|| = 1
# where. ||B||_infty = max sum of entries 

# so what the other methods

# ╔═╡ 3843bcca-0ce9-4cde-8dc1-d911c6ddadad
md"""
!!! danger \"Obstacles for implementing the explicit Euler method\"

**Explicit Euler** methods
- **Conditionally stable**
- only works for very small time steps. .  
"""

# ╔═╡ 26428334-2300-4d1d-8a61-d9d365c26891
#Cause stepsize in time to be dependent on the numerical method we choose
#leading generally to more time steps taken to get to a desired time accuracy. 


# ╔═╡ 76acb6fc-6d94-496d-b70b-c6b746e95a8b
md"""
!!! note \"Discrete Maximum Principle\"

Let us regard the Implicit Euler method ( $\theta = 1$ )

```math
\begin{aligned}
\frac1{\tau^n} \left(M u^n - M u^{n - 1}\right) + A u^{n} &= 0\\
\frac1{\tau^n} M u^n + A u^n  &= \frac1{\tau^n} M u^{n-1}\\
\frac1{\tau^n} m_{kk} u^n_k + a_{kk} u^n_k &= \frac1{\tau^n} m_{kk} u^{n-1}_k+ \sum_{k\neq l} (-a_{kl}) u^n_l \\
u^n_k&\leq     \max (\{u^{n-1}_k\}\cup \{u^n_l\}_{l \in \mathcal N_k})
\end{aligned}
```
Important in this case that we have a positive diagonal matrix $M$ that makes this estimate possible. 
"""

# ╔═╡ 383c85fd-54a7-40cf-942c-0d1a2b269c29
#Another interesting physical property, if we want to prevent oscillation of the solution.

#Indeed, the value of the solution at certain discretization point at new timestep estimated by the solution from the old timestep, and by the solution in the neigboring points.

# ╔═╡ 6e285dce-e986-4787-81fc-6c2e0e950d6d
md"""
!!! note \"Nonnegativity\"
How about the nonnegativity of the solution in our timestep
```math
  \begin{aligned}
    u^n  +  \tau^nM^{-1}Au^n &= u^{n-1}\\
    u^n&= (I+\tau^nM^{-1}A)^{-1}u^{n-1}
  \end{aligned}
```
  
  -  ``(I+\tau^nM^{-1}A)`` is an M-Matrix  that means
  -  If ``u_0>0``, then ``u^n>0`` ``\forall n>0`` 
"""

# ╔═╡ 47852448-03c8-46a8-b9cd-dcd2af3d0907
#and its inverse has positive entries

# ╔═╡ bdb653b8-5b94-4ca1-871a-f80672da478e
md"# Numerical Solution Method

We look at the full discretization which consists of combining the voronoi finite volume and backward Euler methods:

```math
  \begin{aligned}
    u^n  +  \tau^nM^{-1}Au^n &= u^{n-1}\\
    u^n&= (I+\tau^nM^{-1}A)^{-1}u^{n-1}
  \end{aligned}
```


!!! note \"Why Implicit Euler method?\"

- **Stability over a wide range of time steps**
- Number of timesteps is independent of the size of the space discretization.
- We can get very accurate iterative solvers for steady-state problems. 

***
"

# ╔═╡ 8c0217a8-9c21-4a12-b505-e53e4dc87a2a
#We performed space and time discretization of our computational domain. we can now solve the system of our PDE.

#uncond stable ....In particular, without any additional condition, we can guarantee that we have no physical oscillation of our solution.

#Time accuracy is less important and number ....

#It is helpful when stability is more valuable than accuracy in time and then we don’t worry much about the stability of the scheme

# ╔═╡ 942a0641-aafa-45e6-b3aa-4e49bcd4cbe4
md"# Simulation Results

We present in this section the numerical results we obtained for both the 1D and 2D cases. 
"

# ╔═╡ 6c58cdc0-1709-43f9-8136-1565e3526cab
md"## 1D Case
"

# ╔═╡ d25b1c87-18b4-4a4f-9d0c-6116bb337211
md"
#####
We use **VoroinoiFVM.jl** to approximate the solution. We do so by creating the physics (flux and storage) and passing it to the `System` constructor. Additionally, by specifying the `bcondition` callback in the constructor, we can include all three kinds of homogeneous boundary conditions (Dirichlet, Neumann, Robin). The solver 	control uses **Newton method** to solve the nonlinear system of equations. 

For the time interval, $t_0 = 0.001$ and $t_1 = 0.01$ were chosen so that the support of the Barenblatt solution, $b(x,t_1)$, lies in $(-1,1)$. 
"

# ╔═╡ 1f73eb1b-32fe-4719-aa2f-fe8f7165962e
md"""
!!! tip \"Homogeneous Robin BC\"
"""

# ╔═╡ 627de51e-2f41-47b8-8606-9548a7cedc4b
md"""
The error is calculated by comparing the numerical solution to the Barenblatt solution at $t_1 = 0.01$.
"""

# ╔═╡ 37d4b8f7-2a2d-4e5c-b17d-78929860bd9a
md"""
!!! info \"Solution plots\"
"""

# ╔═╡ 308f28ff-8ebd-4abd-965d-a4f65d5532bb
md"""
Binding the timesteps to the slider, we can see the numerical solution plotted against our domain $(-1,1)$ at various points in time. To compare the error, the Barenblatt solution is also plotted on the same graph.
"""

# ╔═╡ b7361bb9-19f6-480a-b31e-6d527af57145
md"""
The following plot shows the evolution of the numerical solution throughout time. We plot the domain on the $x$-axis and the time interval on the $y$-axis, thus representing our entire solution across time.  
"""

# ╔═╡ 09ce90d4-5acd-42c6-9a8e-c9fc55c97e41
md"""
!!! tip \"Homogeneous Neumann BC"
"""

# ╔═╡ 0f9c8cac-5769-4c0b-8396-d6a289a61394
md"""
!!! info \"Solution plots\"
"""

# ╔═╡ 56d95760-14da-4397-aa53-82f67da91bae
md"""
!!! tip \"Homogeneous Dirichlet BC\"
"""

# ╔═╡ eacc9e3b-cac4-4718-a9f0-aa5742f291eb
md"""
!!! info \"Solution plots\"
"""

# ╔═╡ 07f0b050-332f-4405-96ed-c675d61cb2cc
md"""
***
"""

# ╔═╡ ee7647db-3318-4658-a9f7-0e1398a32b45
md"""
!!! tip \"Error of solution vs Grid spacing\"
"""

# ╔═╡ b5c17195-5da8-4411-8916-59321c6f6898
md"""
We manually run our function `diffusion1d` for different values of $N$, the number of grid points. The error in each case is then plotted against the grid spacing.
"""

# ╔═╡ 23e3562b-a650-4c1e-b4c1-0b7bcfa20f0b
md"
!!! tip \"Choosing the time values\"
We need to choose the time values $t_0$ and $t_1$ such that the support of $b(x,t_1)$ is contained in $(-1,1)$. 

i.e.,

$$-1 < - 2t^{\frac{\beta}{2}}\sqrt{\frac{m}{\beta (m - 1)}} \leq x \leq 2t^{\frac{\beta}{2}}\sqrt{\frac{m}{\beta (m - 1)}} < 1$$ which grows as $t$ grows.

To start with, let us implement the Barenblatt solution as functions. 
"

# ╔═╡ 94727561-6df3-4695-ad10-33d28692ac8d
function barenblatt(x,t,m)
	tx=t^(-1.0/(m+1.0))
    xx=x*tx
    xx=xx*xx
    xx=1- xx*(m-1)/(2.0*m*(m+1));
    if xx<0.0
        xx=0.0
    end
    return tx*xx^(1.0/(m-1.0))
end

# ╔═╡ 19fd8cfa-6ab8-4d32-af96-48afd347652b
function barenblatt2d(x,y,t,m)
	dim = 2
	alpha=(1/(m-1+(2/dim)))
	r=sqrt(x^2+y^2)
	K=(alpha*(m-1)*r^2)/(2*dim*m*(t^(2*alpha/dim)))
	
	B=(t^(-alpha))*((1-K)^(1/(m-1)));
	
    if B<0.0
        B=0.0
    end
	
    return B
end

# ╔═╡ dc8ba372-b603-4651-ba85-29664e354d5a
md"
##### 
The support of the Barenblatt solution lies in the domain for $t0 > 0.00001$ and $t1 <0.01$.
"

# ╔═╡ 1fbf8f45-09b6-4b5a-a9d7-073518bed218
md"""
time=$(@bind t1 Slider(0.000001:0.000001:0.015,default=0.0001,show_value=true))
"""

# ╔═╡ 30fc9190-a2a4-4c64-90b4-78ece55af3aa
let
	clf()
	PyPlot.grid()
	N=100
	X=collect(range(-2,2,length=N))
	PyPlot.plot(X,map(x->barenblatt(x,10^(-2),m),X),label=@sprintf("upper time, t1=%.3g",10^(-2)))
	PyPlot.plot(X,map(x->barenblatt(x,t1,m),X), label="time changing")
	PyPlot.plot(X,map(x->barenblatt(x,10^(-5),m),X),label =@sprintf("lower time, t0=%.3g",10^(-5)))
	PyPlot.xlabel("grid in the domain (-1,1)")
	PyPlot.ylabel("exact solution")
	legend()
	gcf().set_size_inches(10,3)
	gcf()
end

# ╔═╡ 94a1a20d-9d3a-4510-ab74-5d7efc6e8519
md"""
## 2D Case
We repeat the above analysis for the 2D case.
"""

# ╔═╡ 9566c847-a50a-4651-8f3e-42c19f3c1cea
md"""
!!! tip \"Homogeneous Robin BC\"
"""

# ╔═╡ bdb1fd67-7340-447e-844a-74b49930dbac
md"""
!!! info \"Solution plot\"
"""

# ╔═╡ 0c7888d5-25c1-4926-ba88-ab8ed2f050e7
md"""
!!! tip \"Homogeneous Neumann B"
"""

# ╔═╡ 32b43db6-9313-4e7b-a91e-00371ad56805
md"""
!!! info \"Solution plot\"
"""

# ╔═╡ e5bc318d-6884-409b-9952-05c1226998ad
md"""
!!! tip \"Homogeneous Dirichlet BC\"
"""

# ╔═╡ 9ce8ac44-c6ca-4f5b-864e-f4073fdefe7c
md"""
!!! info \"Solution plot\"
"""

# ╔═╡ db82964f-84ce-4a8f-8eec-bea773ff8465
md"""
!!! tip \"Error of solution vs Grid spacing\"
"""

# ╔═╡ b1ae1ef6-9751-46f5-9064-40f2858b93b2
md"""
This analysis is repeated for the 2D case, however here we observe the opposite effect.
"""

# ╔═╡ 9da94e44-8165-47a3-8360-4c168cee8aa7
md" # Optional
"

# ╔═╡ 938f262b-0c24-4704-ac1a-45dba9b9ef1f
md" ## Improving Performance
"

# ╔═╡ 1c1d9cd9-6c15-4a39-9de6-d733d8d49e88
md"
Using **DifferentialEquations.jl** for transient problems can yield better performance. For 1D this is indeed the case, as the **DifferentialEquations.jl** solver runs faster and also produces less error. In the 2D case however, the runtime is slower than that of VoronoiFVM.jl while the error is still lower.
"

# ╔═╡ b16b1c5f-29df-4abf-becf-6f40270bf697
md"""
### 1D Case
"""

# ╔═╡ 8ea27a48-667a-4b11-999c-754351df4ca9
md"""
### 2D Case
"""

# ╔═╡ 8bd9734e-0a11-4172-82b6-eb168e2eeec8
md"""
# References
Listed below are the references we used for this project (URLs included).

1. Chowdhury, Atiqur & Barmon, Ashish & Alam, Sharmin & Akter, Maria. (2017). [Numerical Simulation that Provides Non-oscillatory Solutions for Porous Medium Equation and Error Approximation of Boussinesq's Equation](https://www.researchgate.net/publication/314114609_Numerical_Simulation_that_Provides_Non-oscillatory_Solutions_for_Porous_Medium_Equation_and_Error_Approximation_of_Boussinesq%27s_Equation), Universal Journal of Computational Mathematics.  

2. Vazquez, Juan Luis. (2007). [The Porous Medium Equation: Mathematical Theory](https://www.researchgate.net/publication/314114609_Numerical_Simulation_that_Provides_Non-oscillatory_Solutions_for_Porous_Medium_Equation_and_Error_Approximation_of_Boussinesq%27s_Equation). 

3. Wu, Zhuoqun & Yin, Jingxue & Li, Huilai & Zhao, Junning. (2001). [Nonlinear Diffusion Equations.](https://www.researchgate.net/publication/344083045_Nonlinear_Diffusion_Equations)

4. Knabner, Pter & Angermann, Lutz. (2003). [Numerical Methods for Elliptic and Parabolic Partial Differential Equations.](https://link.springer.com/book/10.1007/b97419), Springer-Verlag.
"""

# ╔═╡ 6c3cc81e-e6f9-4178-aa78-b2d66ad8662b
md"""
# Julia functions
"""

# ╔═╡ 873ce4e6-d9bc-4abb-937e-2089660ab819
function robin_bc!(f,u,bnode)
	VoronoiFVM.boundary_robin!(f,u,bnode,factor=1,value=0)
end

# ╔═╡ 2fe891c5-453c-4c46-9312-cc67adf9ed8e
function dirichlet_bc!(f,u,bnode)
    VoronoiFVM.boundary_dirichlet!(f,u,bnode,value=0)
end

# ╔═╡ d45e266f-70e0-4cc7-a53c-7bd8be691855
function neumann_bc!(f,u,bnode)
	VoronoiFVM.boundary_neumann!(f,u,bnode,value=0)
end

# ╔═╡ 83b59fc1-1e0c-44b6-b345-ccb5275f2d56
function diffusion1d(m,N=N11,t0 = 0.001,tend=0.01,tstep=1.0e-6;bcond=neumann_bc!)

	x=collect(range(-1,1,length=N))
	grid1d=simplexgrid(x)
	
	## Define flux between neighbouring control volumes,
	## in our case D*(u_k - u_l) = u_k^m - u_l^m
	function flux!(f,u,edge)
        f[1]=u[1,1]^m-u[1,2]^m
    end

	## Define the term under time derivative
	function storage!(f,u,node)
		f[1]=u[1]
	end

	## Define system 
	
	sys=VoronoiFVM.System(grid1d,flux=flux!,storage=storage!,species=[1],bcondition=bcond)
	enable_species!(sys,1,[1])
	## Initial value should be broadcasted to b(x,t0)
	inival=unknowns(sys)
	inival[1,:].=map(x->barenblatt(x,t0,m),grid1d)

	## Initialize solver control, set parameters
	control=VoronoiFVM.NewtonControl()
	control.Δt_min=tstep
	control.Δt=tstep
	control.Δt_max=tstep

	## Solving the system and calculating the error 
	tsol=VoronoiFVM.solve(sys,inival=inival,times=[t0,tend];control=control)
	err=norm(tsol[1,:,end]-map(x->barenblatt(x,tend,m),grid1d))
	return tsol,err
	
end

# ╔═╡ 1d66f531-318b-4746-9d73-59b11c6897a9
tsol_robin,err_robin = diffusion1d(m, bcond=robin_bc!);

# ╔═╡ 56eeb308-6c35-4fb2-a80b-335d9ff4fc2f
err_robin

# ╔═╡ 4b6163ae-f7ac-40cd-a97e-12184a08537f
md"""
Timestep: $(@bind τ_robin Slider(1:length(tsol_robin),default=1,show_value=true))
"""

# ╔═╡ 3d8245d0-1b45-4447-bb91-445ba8db0713
md"""
Time: $(tsol_robin.t[τ_robin])
"""

# ╔═╡ 5ea7537b-dcfe-4d9c-a614-9d16fdd6dc92
sol_robin_τ=tsol_robin[1,:,τ_robin]

# ╔═╡ ef3fce87-e0a2-4f8a-b8a8-b6f92ecfa553
let
    p_robin=PlutoVistaPlot(resolution=(500,300),titlefontsize=20)
	PlutoVista.plot!(p_robin,X17,sol_robin_τ;label="Numerical Solution",color=:red,linestyle=:dashdot)
	PlutoVista.plot!(p_robin,X17,map(x->barenblatt(x,tsol_robin.t[τ_robin],m), X17);label="Barenblatt Solution",color=:green,linewidth=1,markertype=:star5,legend=:rt)
end

# ╔═╡ ea1b9dc7-e9b9-4d37-a1ca-dd2651014ca7
tsol,err = diffusion1d(m);

# ╔═╡ 94ca5b00-cb9c-4fa8-86d3-fee8d7075646
err

# ╔═╡ 13b62e49-d4e2-4cf9-8796-2f0484da892d
md"""
Timestep: $(@bind τ Slider(1:length(tsol),default=1,show_value=true))
"""

# ╔═╡ bfedc9b1-c86c-48c8-9a7a-0f3ace4750b5
md"""
Time: $(tsol.t[τ])
"""

# ╔═╡ a6e016c6-5682-4c1d-bfb7-818a5f71fd88
sol_τ=tsol[1,:,τ]

# ╔═╡ 2755af91-f471-4ab7-89ef-8792c7ddaa01
let
    p=PlutoVistaPlot(resolution=(500,300),titlefontsize=20)
	PlutoVista.plot!(p,X17,sol_τ;label="Numerical Solution",color=:red,linestyle=:dashdot,markertype=:circle)
	PlutoVista.plot!(p,X17,map(x->barenblatt(x,tsol.t[τ],m), X17);label="Barenblatt Solution",color=:green,linewidth=1,markertype=:star5,legend=:rt
	)
end

# ╔═╡ 6a459adf-1643-4c1e-922f-ade2b2d9299f
tsol_dirichlet,err_dirichlet=diffusion1d(m, bcond=dirichlet_bc!);

# ╔═╡ ba95a518-3f5d-46c3-a61d-68ed9226c779
err_dirichlet

# ╔═╡ 22612c7a-2c8b-487b-9c02-4650c9702e51
md"""
Timestep: $(@bind τ_dirichlet Slider(1:length(tsol_dirichlet),default=1,show_value=true))
"""

# ╔═╡ e8f4f6fb-b17d-4812-ade6-c5e0b2a0be1c
md"""
Time: $(tsol_dirichlet.t[τ_dirichlet])
"""

# ╔═╡ b127d0ce-cbe1-4850-8f87-5321eabe33da
sol_dirichlet_τ=tsol_dirichlet[1,:,τ_dirichlet]

# ╔═╡ 30cad126-d541-4b19-8a78-69cc888ceb64
let
    p_dirichlet=PlutoVistaPlot(resolution=(500,300),titlefontsize=20)
	PlutoVista.plot!(p_dirichlet,X17,sol_dirichlet_τ;label="Numerical Solution",color=:red,linestyle=:dashdot)
	PlutoVista.plot!(p_dirichlet,X17,map(x->barenblatt(x,tsol_dirichlet.t[τ_dirichlet],m), X17);label="Barenblatt Solution",color=:green,linewidth=1,markertype=:star5,legend=:rt)
end

# ╔═╡ 5b44c6b1-536b-4c97-954a-26f0a3bbe2b6
t=@elapsed begin
	sol_test,err_test=diffusion1d(m)
end

# ╔═╡ 10772ca8-042a-4f3b-8ad3-9052fbf0dcfc
function plotsol(sol,X)
	f=sol[1,:,:]'
	PyPlot.contourf(X,sol.t,f,0:0.1:10,cmap=:summer)
	PyPlot.contour(X,sol.t,f,0:1:10,colors=:black)
end

# ╔═╡ 74e96e04-65cd-4c85-84e4-f8560ac4bc9b
PyPlot.clf();plotsol(tsol_robin,X17); gcf()

# ╔═╡ fdc138e7-3e9f-4bef-a56a-3f613f0779a4
PyPlot.clf();plotsol(tsol,X17); gcf()

# ╔═╡ dd4627cb-8b30-4bf1-af4f-8be1b04d198a
PyPlot.clf();plotsol(tsol_dirichlet,X17); gcf()

# ╔═╡ ed60becf-7757-4952-bfff-6916a299e79b
function diffusion2d(m,N=N13,t0 = 0.001,tend=0.01,tstep=1.0e-6,grid2d=grid2d;bcond=neumann_bc!)

	x=collect(range(-1,1,length=N))
	grid2d=simplexgrid(x,x)
	
	function flux!(f,u,edge)
        f[1]=u[1,1]^m-u[1,2]^m
    end
	
    function storage!(f,u,node)
        f[1]=u[1]
    end
	
    sys2d=VoronoiFVM.System(grid2d,flux=flux!,storage=storage!,species=[1],bcondition=bcond)
	
	
	inival=unknowns(sys2d)
	inival[1,:].=map((x,y)->barenblatt2d(x,y,t0,m),grid2d)
			
	control=VoronoiFVM.NewtonControl()
	control.Δt_min=0.01*tstep
	control.Δt=tstep
	control.Δt_max=0.1*tend
	control.Δu_opt=0.8

	tsol2d=VoronoiFVM.solve(sys2d,inival=inival,times=[t0,tend];control=control)
	err2d=norm(tsol2d[1,:,end]-map((x,y)->barenblatt2d(x,y,tend,m),grid2d))
	return tsol2d,err2d
	
end

# ╔═╡ ffe2bfb9-8f22-4241-8264-6bde8b5c89c4
tsol2d_robin,err2d_robin=diffusion2d(m, bcond=robin_bc!);

# ╔═╡ 620cca8c-1d77-4bd8-9078-9844141ba7bf
err2d_robin

# ╔═╡ bbf20d7a-fa2a-45b7-950f-b2930e3452e4
md"""
Timestep: $(@bind τ2d_robin Slider(1:length(tsol2d_robin),default=1,show_value=true))
"""

# ╔═╡ 265871ac-edfc-4673-9c16-8ff28436c83b
md"""
Time: $(tsol2d_robin.t[τ2d_robin])
"""

# ╔═╡ 5eace558-427d-4c9a-84d3-46cb0650e8d5
sol_τ2d_robin=tsol2d_robin[1,:,τ2d_robin]

# ╔═╡ d9a956f5-c36a-4e11-ada5-25b1478be63f
begin
	p2d_robin=GridVisualizer(;Plotter=PyPlot,layout=(1,2),clear=true,resolution=(900,450))
	scalarplot!(p2d_robin[1,1],grid2d,tsol2d_robin[1,:,τ2d_robin],title="Numerical solution")
	scalarplot!(p2d_robin[1,2],grid2d,map((x,y)->barenblatt2d(x,y,tsol2d_robin.t[τ2d_robin],m), grid2d),title="Exact solution")
	reveal(p2d_robin)
end

# ╔═╡ 88638b27-0bd8-4829-a8fa-90adebc6bca6
tsol2d,err2d=diffusion2d(m);

# ╔═╡ bbf25920-5681-4099-8a92-a92e33f429e2
err2d

# ╔═╡ 2850e93a-f30c-43e6-99e1-ab44cfcf9241
md"""
Timestep: $(@bind τ2d Slider(1:length(tsol2d),default=1,show_value=true))
"""

# ╔═╡ be810e4c-d3a8-4629-ac17-3285ca1a9026
md"""
Time: $(tsol2d.t[τ2d])
"""

# ╔═╡ 2c9e5ac3-996d-4e20-b192-c0b10884c7f9
sol_τ2d=tsol2d[1,:,τ2d]

# ╔═╡ f02c1eac-1ca2-4ac8-ad3d-522dd5e0312e
begin
	p2d=GridVisualizer(;Plotter=PyPlot,layout=(1,2),resolution=(900,450))
	scalarplot!(p2d[1,1],grid2d,tsol2d[1,:,τ2d],title="Numerical solution")
	scalarplot!(p2d[1,2],grid2d,map((x,y)->barenblatt2d(x,y,tsol2d.t[τ2d],m), grid2d),title="Exact solution")
	reveal(p2d)
end

# ╔═╡ ffe2659e-60f3-42b9-8b77-f39edb276222
tsol2d_dirichlet,err2d_dirichlet=diffusion2d(m, bcond=dirichlet_bc!);

# ╔═╡ 14e06383-ef26-48be-ad72-eb80b6418dfd
err2d_dirichlet

# ╔═╡ 3a9b42f2-a3d5-468d-b1b5-9cf0f281e4e2
md"""
Timestep: $(@bind τ2d_dirichlet Slider(1:length(tsol2d_dirichlet),default=1,show_value=true))
"""

# ╔═╡ df2bd4ac-d669-43b2-9ad3-443846542427
md"""
Time: $(tsol2d_dirichlet.t[τ2d_dirichlet])
"""

# ╔═╡ 9dbc3570-9d40-4147-98fb-bc9659b71d50
sol_τ2d_dirichlet=tsol2d_dirichlet[1,:,τ2d_dirichlet]

# ╔═╡ 64c72c4e-a879-46b1-b34e-3286fbd6ad23
begin
	p2d_dirichlet=GridVisualizer(;Plotter=PyPlot,layout=(1,2),clear=true,resolution=(900,450))
	scalarplot!(p2d_dirichlet[1,1],grid2d,tsol2d_dirichlet[1,:,τ2d_dirichlet],title="Numerical solution")
	scalarplot!(p2d_dirichlet[1,2],grid2d,map((x,y)->barenblatt2d(x,y,tsol2d_dirichlet.t[τ2d_dirichlet],m), grid2d),title="Exact solution")
	reveal(p2d_dirichlet)
end

# ╔═╡ da947fef-2e06-4b74-b7e1-95f09086bbc1
t3=@elapsed begin
	sol_test2d,err_test2d=diffusion2d(m)
end

# ╔═╡ 453953fb-674a-4d9c-9aa0-43ec73be155d
function err_sol_vs_grid_spacing(m,bcond=neumann_bc!)
	Nlist=collect(10:10:100)
	spacinglist=[]
	errlist=[]
	for N in Nlist
		spacing=2.0/(N-1)
		append!(spacinglist,spacing)
		tsol_N,err_N=diffusion1d(m,N,bcond=bcond)
		append!(errlist, err_N)
	end
	return spacinglist,errlist
end	

# ╔═╡ 7da5e1d7-2ec6-41aa-9cb1-ea15457130d4
spacinglist,errlist=err_sol_vs_grid_spacing(m)

# ╔═╡ 70d71a37-aa11-4a33-a0c1-aeca87a30f3a
PlutoVista.plot(spacinglist,errlist,resolution=(450,450),title="Error of solution vs Grid spacing",titlefontsize=15,xlabel="Grid spacing",ylabel="Error of solution")

# ╔═╡ 1f07b937-b066-4298-b0ba-5bab59f79b8b
function err_sol_vs_grid_spacing2d(m,bcond=neumann_bc!)
	Nlist2d=collect(10:10:100)
	spacinglist2d=[]
	errlist2d=[]
	for N in Nlist2d
		spacing2d=2.0/(N-1)
		append!(spacinglist2d,spacing2d)
		tsol_N2d,err_N2d=diffusion2d(m,N,bcond=bcond)
		append!(errlist2d, err_N2d)
	end
	return spacinglist2d,errlist2d
end	

# ╔═╡ 61584f0d-84e7-4885-ab6c-f519128a68a7
spacinglist2d,errlist2d=err_sol_vs_grid_spacing2d(m)

# ╔═╡ 0d1e382f-992b-457a-afcd-249b303b6617
PlutoVista.plot(spacinglist2d,errlist2d,resolution=(450,450),title="Error of solution vs Grid spacing",titlefontsize=15,xlabel="Grid spacing",ylabel="Error of solution")

# ╔═╡ f01c7a8f-3386-46d6-96f6-dece75b46836
function diffusion1d_perf(m,N=N11,t0 = 0.001,tend=0.01,tstep=1.0e-6;bcond=neumann_bc!)

	x=collect(range(-1,1,length=N))
	grid1d=simplexgrid(x)
	
	## Define flux between neighbouring control volumes,
	## in our case D*(u_k - u_l) = u_k^m - u_l^m
	function flux!(f,u,edge)
        f[1]=u[1,1]^m-u[1,2]^m
    end

	## Define the term under time derivative
	function storage!(f,u,node)
		f[1]=u[1]
	end

	## Define system 
	sys=VoronoiFVM.System(grid1d,flux=flux!,storage=storage!,species=[1],bcondition=bcond)
	enable_species!(sys,1,[1])
	
	## Initial value should be broadcasted to b(x,t0)
	inival=unknowns(sys)
	inival[1,:].=map(x->barenblatt(x,t0,m),grid1d)

	## Initialize solver control, set parameters
	control=VoronoiFVM.NewtonControl()
	control.Δt_min=0.01*tstep
	control.Δt=tstep
	control.Δt_max=0.1*tend

	## Solving the system and calculating the error 
	problem = ODEProblem(sys,inival,(t0,tend))
	odesol = DifferentialEquations.solve(problem)
	tsol=reshape(odesol,sys)
	err=norm(tsol[1,:,end]-map(x->barenblatt(x,tend,m),grid1d))
	return tsol,err
	
end

# ╔═╡ e252b6a3-b840-4a18-b085-1203deb52674
t2=@elapsed begin
	sol_test2,err_test2=diffusion1d_perf(m)
end

# ╔═╡ 97566eca-a486-46b5-91a3-0f5c27b07e8b
begin
	perf_plot=GridVisualizer(;Plotter=PyPlot,layout=(1,2),clear=true,resolution=(900,450))
	scalarplot!(perf_plot[1,1],grid1d,sol_test[1,:,end],title=@sprintf("VoronoiFVM: %.0f ms e=%.2e",t*1000,err_test))
	scalarplot!(perf_plot[1,2],grid1d,sol_test2[1,:,end],title=@sprintf("DifferentialEq: %.0f ms, e=%.2e",t2*1000,err_test2))
	reveal(perf_plot)
end

# ╔═╡ 12cde0a8-4e38-4e5d-b3fe-90f24763d9cb
function diffusion2d_perf(m,N=N13,t0 = 0.001,tend=0.01,tstep=1.0e-6;bcond=neumann_bc!)
	
	x=collect(range(-1,1,length=N))
	grid2d=simplexgrid(x,x)
	
	function flux!(f,u0,edge)
        u=unknowns(edge,u0)
        f[1]=u[1,1]^m-u[1,2]^m
    end
	
    function storage!(f,u,node)
        f[1]=u[1]
    end
	
    sys2d=VoronoiFVM.System(grid2d,flux=flux!,storage=storage!,species=[1],bcondition=bcond)
	
	
	inival=unknowns(sys2d)
	inival[1,:].=map((x,y)->barenblatt2d(x,y,t0,m),grid2d)
			
	control=VoronoiFVM.NewtonControl()
	control.Δt_min=0.01*tstep
	control.Δt=tstep
	control.Δt_max=0.1*tend
	control.Δu_opt=0.8

	problem2d = ODEProblem(sys2d,inival,(t0,tend))
	odesol2d = DifferentialEquations.solve(problem2d)
	tsol2d=reshape(odesol2d,sys2d)
	err2d=norm(tsol2d[1,:,end]-map((x,y)->barenblatt2d(x,y,tend,m),grid2d))
	return tsol2d,err2d
	
end

# ╔═╡ aef75228-fc8a-4944-b7cb-067f2f31c8bd
t4=@elapsed begin
	sol_test2d2,err_test2d2=diffusion2d_perf(m)
end

# ╔═╡ a21486a5-5233-4131-b785-3358fc0d43fc
begin
	perf_plot2d=GridVisualizer(;Plotter=PyPlot,layout=(1,2),clear=true,resolution=(900,450))
	scalarplot!(perf_plot2d[1,1],grid2d,sol_test2d[1,:,end],title=@sprintf("VoronoiFVM: %.0f ms e=%.2e",t3*1000,err_test2d))
	scalarplot!(perf_plot2d[1,2],grid2d,sol_test2d2[1,:,end],title=@sprintf("DifferentialEq: %.0f ms, e=%.2e",t4*1000,err_test2d2))
	reveal(perf_plot)
end

# ╔═╡ 1664b848-906c-43a7-879e-270ee283e6f3
PlutoUI.TableOfContents()

# ╔═╡ Cell order:
# ╟─281bdc34-664f-4aaf-83ec-53e89cfdb4d8
# ╟─e112e366-9463-464d-b878-699475ee48a5
# ╠═4d88b926-9543-11ea-293a-1379b1b5ae64
# ╠═c46006d6-085d-487c-a7cd-91b8126fb960
# ╟─76a630d5-f675-4e51-9b7a-ee357264e9f6
# ╠═3c3a3d5f-17c2-4f37-8b3e-4ded02650431
# ╟─125329a8-530c-41ff-83c2-ea27e29891e7
# ╠═9d89499f-ea4d-4548-9b30-e0f8f0a02ca9
# ╟─d47e0e73-8909-4dad-b9e7-380c63033298
# ╠═96c4b528-273a-43e0-8256-9e2ab0bb59ef
# ╟─ed2910e2-9f07-4024-af44-e360a7f831db
# ╟─96bae2ce-cc36-414d-8bcd-5ddd4b88a06a
# ╠═ceb9f9e2-5040-477b-a952-78c1015832a5
# ╟─6a24a9a6-c522-4489-9cc8-75fe3f23a5b0
# ╠═8aac817b-51e5-4b6a-b0f3-4a4c0e4ee42a
# ╠═78c42926-8a64-4244-8f4e-5d6aad7dc19d
# ╟─9392a40c-2a12-438a-906e-2ac036599259
# ╠═ebe0baeb-6ebd-4573-91cd-281d3b0de9d7
# ╟─b914b3da-d35e-49d9-aa02-6468508069f5
# ╠═d521b91b-d2f4-4737-9c32-b8f6b7a0b05d
# ╟─66a6e161-6213-4d91-8750-9a915c827e3a
# ╟─4fee3262-38db-4c23-bf6b-4eddf96c1f02
# ╠═668cd97f-1471-4007-adb4-c02bd4b910db
# ╟─58049525-8458-4575-93aa-6ae35e26f49e
# ╠═f0a06333-19b0-4885-80c8-1687cdc83fb0
# ╟─b356d4c8-5db1-4ca0-8994-7e31c90b3d68
# ╠═efa8ac51-9ab6-4330-b08f-065466b86f08
# ╟─3c023b94-7386-43f5-8234-9ac20a23826c
# ╠═f625420f-fb2c-4ee2-86a3-05ed3072cf42
# ╟─49cbc1f3-0141-4692-b347-41817322982f
# ╟─5ba7370a-3bed-4e85-99dc-f26cfe5e280c
# ╟─1873c8cf-cdf0-4281-955f-43fe6de92d3f
# ╠═5444d294-ffb6-4e07-a9b5-c7ee6fae2d19
# ╠═09ea7716-0027-4974-bff6-8e86e3918365
# ╟─c87d6b95-7a60-4a10-a009-f4421a3d1d65
# ╠═375770f1-f755-4970-bd5d-f273a2a84bd4
# ╟─21d9613f-91d7-4570-aa8c-cfb8871bf8e3
# ╟─eebd876d-ddd6-4570-a100-d497456abd52
# ╠═5add94c0-41a8-4939-ac2a-97df64a257be
# ╟─19b8c750-f80e-4b4a-abea-3a6181bad356
# ╠═065c038c-b33f-4636-ab12-39ec4a5e4096
# ╟─f6ad658a-4d49-479c-b63d-6f2849c1d926
# ╟─fd3d5770-4835-4501-b727-3acdcfbab623
# ╠═04546bf2-f323-4cf8-ad9b-7b3bc72093ce
# ╟─3843bcca-0ce9-4cde-8dc1-d911c6ddadad
# ╠═26428334-2300-4d1d-8a61-d9d365c26891
# ╟─76acb6fc-6d94-496d-b70b-c6b746e95a8b
# ╠═383c85fd-54a7-40cf-942c-0d1a2b269c29
# ╟─6e285dce-e986-4787-81fc-6c2e0e950d6d
# ╠═47852448-03c8-46a8-b9cd-dcd2af3d0907
# ╟─bdb653b8-5b94-4ca1-871a-f80672da478e
# ╠═8c0217a8-9c21-4a12-b505-e53e4dc87a2a
# ╟─942a0641-aafa-45e6-b3aa-4e49bcd4cbe4
# ╟─6c58cdc0-1709-43f9-8136-1565e3526cab
# ╟─d25b1c87-18b4-4a4f-9d0c-6116bb337211
# ╟─1f73eb1b-32fe-4719-aa2f-fe8f7165962e
# ╠═1d66f531-318b-4746-9d73-59b11c6897a9
# ╟─627de51e-2f41-47b8-8606-9548a7cedc4b
# ╠═56eeb308-6c35-4fb2-a80b-335d9ff4fc2f
# ╟─37d4b8f7-2a2d-4e5c-b17d-78929860bd9a
# ╟─308f28ff-8ebd-4abd-965d-a4f65d5532bb
# ╟─4b6163ae-f7ac-40cd-a97e-12184a08537f
# ╟─3d8245d0-1b45-4447-bb91-445ba8db0713
# ╠═5ea7537b-dcfe-4d9c-a614-9d16fdd6dc92
# ╟─ef3fce87-e0a2-4f8a-b8a8-b6f92ecfa553
# ╟─b7361bb9-19f6-480a-b31e-6d527af57145
# ╟─74e96e04-65cd-4c85-84e4-f8560ac4bc9b
# ╟─09ce90d4-5acd-42c6-9a8e-c9fc55c97e41
# ╠═ea1b9dc7-e9b9-4d37-a1ca-dd2651014ca7
# ╠═94ca5b00-cb9c-4fa8-86d3-fee8d7075646
# ╟─0f9c8cac-5769-4c0b-8396-d6a289a61394
# ╟─13b62e49-d4e2-4cf9-8796-2f0484da892d
# ╟─bfedc9b1-c86c-48c8-9a7a-0f3ace4750b5
# ╟─a6e016c6-5682-4c1d-bfb7-818a5f71fd88
# ╟─c91b6729-809c-47da-b47e-0bf41808230b
# ╟─2755af91-f471-4ab7-89ef-8792c7ddaa01
# ╟─fdc138e7-3e9f-4bef-a56a-3f613f0779a4
# ╟─56d95760-14da-4397-aa53-82f67da91bae
# ╠═6a459adf-1643-4c1e-922f-ade2b2d9299f
# ╠═ba95a518-3f5d-46c3-a61d-68ed9226c779
# ╟─eacc9e3b-cac4-4718-a9f0-aa5742f291eb
# ╟─22612c7a-2c8b-487b-9c02-4650c9702e51
# ╟─e8f4f6fb-b17d-4812-ade6-c5e0b2a0be1c
# ╟─b127d0ce-cbe1-4850-8f87-5321eabe33da
# ╟─30cad126-d541-4b19-8a78-69cc888ceb64
# ╟─dd4627cb-8b30-4bf1-af4f-8be1b04d198a
# ╟─07f0b050-332f-4405-96ed-c675d61cb2cc
# ╟─ee7647db-3318-4658-a9f7-0e1398a32b45
# ╟─b5c17195-5da8-4411-8916-59321c6f6898
# ╠═7da5e1d7-2ec6-41aa-9cb1-ea15457130d4
# ╟─70d71a37-aa11-4a33-a0c1-aeca87a30f3a
# ╟─23e3562b-a650-4c1e-b4c1-0b7bcfa20f0b
# ╟─94727561-6df3-4695-ad10-33d28692ac8d
# ╟─19fd8cfa-6ab8-4d32-af96-48afd347652b
# ╟─dc8ba372-b603-4651-ba85-29664e354d5a
# ╟─1fbf8f45-09b6-4b5a-a9d7-073518bed218
# ╟─30fc9190-a2a4-4c64-90b4-78ece55af3aa
# ╟─94a1a20d-9d3a-4510-ab74-5d7efc6e8519
# ╟─9566c847-a50a-4651-8f3e-42c19f3c1cea
# ╠═ffe2bfb9-8f22-4241-8264-6bde8b5c89c4
# ╠═620cca8c-1d77-4bd8-9078-9844141ba7bf
# ╟─bdb1fd67-7340-447e-844a-74b49930dbac
# ╟─bbf20d7a-fa2a-45b7-950f-b2930e3452e4
# ╟─265871ac-edfc-4673-9c16-8ff28436c83b
# ╟─5eace558-427d-4c9a-84d3-46cb0650e8d5
# ╟─d9a956f5-c36a-4e11-ada5-25b1478be63f
# ╟─0c7888d5-25c1-4926-ba88-ab8ed2f050e7
# ╠═88638b27-0bd8-4829-a8fa-90adebc6bca6
# ╠═bbf25920-5681-4099-8a92-a92e33f429e2
# ╟─32b43db6-9313-4e7b-a91e-00371ad56805
# ╟─2850e93a-f30c-43e6-99e1-ab44cfcf9241
# ╟─be810e4c-d3a8-4629-ac17-3285ca1a9026
# ╟─2c9e5ac3-996d-4e20-b192-c0b10884c7f9
# ╟─f02c1eac-1ca2-4ac8-ad3d-522dd5e0312e
# ╟─e5bc318d-6884-409b-9952-05c1226998ad
# ╠═ffe2659e-60f3-42b9-8b77-f39edb276222
# ╠═14e06383-ef26-48be-ad72-eb80b6418dfd
# ╟─9ce8ac44-c6ca-4f5b-864e-f4073fdefe7c
# ╟─3a9b42f2-a3d5-468d-b1b5-9cf0f281e4e2
# ╟─df2bd4ac-d669-43b2-9ad3-443846542427
# ╟─9dbc3570-9d40-4147-98fb-bc9659b71d50
# ╟─64c72c4e-a879-46b1-b34e-3286fbd6ad23
# ╟─db82964f-84ce-4a8f-8eec-bea773ff8465
# ╟─b1ae1ef6-9751-46f5-9064-40f2858b93b2
# ╠═61584f0d-84e7-4885-ab6c-f519128a68a7
# ╟─0d1e382f-992b-457a-afcd-249b303b6617
# ╟─9da94e44-8165-47a3-8360-4c168cee8aa7
# ╟─938f262b-0c24-4704-ac1a-45dba9b9ef1f
# ╟─1c1d9cd9-6c15-4a39-9de6-d733d8d49e88
# ╟─b16b1c5f-29df-4abf-becf-6f40270bf697
# ╠═5b44c6b1-536b-4c97-954a-26f0a3bbe2b6
# ╠═e252b6a3-b840-4a18-b085-1203deb52674
# ╟─97566eca-a486-46b5-91a3-0f5c27b07e8b
# ╟─8ea27a48-667a-4b11-999c-754351df4ca9
# ╠═da947fef-2e06-4b74-b7e1-95f09086bbc1
# ╠═aef75228-fc8a-4944-b7cb-067f2f31c8bd
# ╟─a21486a5-5233-4131-b785-3358fc0d43fc
# ╟─8bd9734e-0a11-4172-82b6-eb168e2eeec8
# ╟─6c3cc81e-e6f9-4178-aa78-b2d66ad8662b
# ╟─83b59fc1-1e0c-44b6-b345-ccb5275f2d56
# ╟─873ce4e6-d9bc-4abb-937e-2089660ab819
# ╟─2fe891c5-453c-4c46-9312-cc67adf9ed8e
# ╟─d45e266f-70e0-4cc7-a53c-7bd8be691855
# ╟─10772ca8-042a-4f3b-8ad3-9052fbf0dcfc
# ╟─ed60becf-7757-4952-bfff-6916a299e79b
# ╟─453953fb-674a-4d9c-9aa0-43ec73be155d
# ╟─1f07b937-b066-4298-b0ba-5bab59f79b8b
# ╟─f01c7a8f-3386-46d6-96f6-dece75b46836
# ╟─12cde0a8-4e38-4e5d-b3fe-90f24763d9cb
# ╟─07f7fd76-c0eb-42b1-843e-0ff67138070a
# ╟─9e67d263-3662-4fb0-9054-2afd3d38b851
# ╟─215aa828-34aa-42f9-a0e0-0e7a46c81cd9
# ╟─1664b848-906c-43a7-879e-270ee283e6f3
