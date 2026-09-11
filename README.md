# Mechanistic Nonlinear Dynamical Modeling for Parameter Estimation: A Comparison of Physics Informed Neural Networks and Genetic Algorithms

This repository contains the computational implementations, manuscript simulations, and robustness/validation analyses associated with the study:

**Mechanistic Nonlinear Dynamical Modeling for Parameter Estimation: A Comparison of Physics Informed Neural Networks and Genetic Algorithms**

**Paul A. Valle¹, Yolocuauhtli Salazar², Jesus B. Páez-Lerma³, N. Oscar Soto-Cruz³, Luis N. Coria¹, Iván A. García¹, Michell V. González-Campos³**

¹ Posgrado en Ciencias de la Ingeniería, BioMath Research Group, Tecnológico Nacional de México/IT Tijuana, Tijuana, México  
² Departamento de Ingeniería Eléctrica y Electrónica, Tecnológico Nacional de México/IT Durango, Durango, México  
³ Departamento de Ingenierías Química y Bioquímica, Tecnológico Nacional de México/IT Durango, Durango, México

**Contact:**  
Paul A. Valle — paul.valle@tectijuana.edu.mx  
Luis N. Coria — luis.coria@tectijuana.edu.mx  
Iván A. García — D25210003@tectijuana.edu.mx  
Yolocuauhtli Salazar — ysalazar@itdurango.edu.mx  
Jesus B. Páez-Lerma — jpaez@itdurango.edu.mx  
N. Oscar Soto-Cruz — nsoto@itdurango.edu.mx  

---

## Overview

The repository accompanies a mechanistic inverse-problem study of batch alcoholic fermentation by *Saccharomyces cerevisiae* ITD00185. The model describes the coupled dynamics of biomass, glucose, fructose, and ethanol, while an accumulated-biomass state introduces history-dependent inhibition.

Parameter estimation is performed using three complementary approaches:

- **Nonlinear Least Squares (LS)** as a classical inverse-problem baseline.
- **Genetic Algorithm (GA)** using a real-coded global search followed by local refinement.
- **Physics-Informed Neural Network (PINN)** using a continuous-time neural representation constrained by experimental data, initial conditions, and the governing differential equations.

The repository also contains the computational analyses used to assess robustness, forecasting performance, sensitivity to raw experimental data, numerical solver dependence, PINN loss weighting, and stochastic initialization.

---

## Mechanistic model

The fermentation model is written as the autonomous ODE system

$$
\frac{dw}{dt} = \left(\rho_1 + \rho_3\frac{x+y}{x_0+y_0} \right) w e^{-\rho_2 W} - \rho_4 w,
$$

$$
\frac{dW}{dt} = w,
$$

$$
\frac{dx}{dt} = -\rho_5 xw-\rho_6 x, $$

$$
\frac{dy}{dt} = -\rho_7 yw-\rho_8 y,
$$

$$
\frac{dz}{dt} = \rho_9(x+y)w-\rho_{10}z.
$$

Here, $w(t)$ is biomass, $W(t)$ is accumulated biomass, $x(t)$ is glucose, $y(t)$ is fructose, and $z(t)$ is ethanol.

The inverse problem estimates

$$
\{\rho_1,\rho_2,\rho_3,\rho_5,\rho_7,\rho_9\},
$$

while $\rho_4$, $\rho_6$, $\rho_8$, and $\rho_{10}$ are fixed according to the mechanistic assumptions described in the manuscript.

The experimental study contains ten fermentation conditions spanning initial total hexose concentrations of **20–100 g/L** and agitation rates of **100–150 rpm**.

## Main computational settings

### PINN

| Setting | Value |
|---|---|
| Framework | TensorFlow |
| Python | 3.10 |
| Architecture | $(1,128,128,128,5)$ |
| Hidden activation | `tanh` |
| Learning rate | $10^{-4}$ |
| Physics/data weight | $\alpha=0.999$ |
| Initial-condition weight | $\beta_{\mathrm{IC}}=0.5$ |
| Physics collocation points | 600 |
| Maximum epochs | 100001 |
| Base random seed | 130425 |
| Estimated kinetic parameters | $\rho_1,\rho_2,\rho_3,\rho_5,\rho_7,\rho_9$ |

The PINN loss combines physics residuals, experimental-data mismatch, and initial-condition constraints. The `Alpha_Sensitivity` tests reproduce the loss-weighting analysis for

$$
\alpha\in\{0.1,0.3,0.5,0.7,0.9,0.95,0.99,0.999,0.9999,1.0\}.
$$

### Genetic Algorithm

| Setting | Value |
|---|---|
| Population size | 300 |
| Maximum generations | 10001 |
| Elite fraction | 0.08 |
| Crossover rate | 0.90 |
| Mutation rate | 0.20 |
| Mutation scale | 0.10 |
| Tournament size | 3 |
| BLX-$\alpha$ | 0.35 |
| Local refinement | `fmincon` / SQP |
| Base random seed | 130425 |
| Estimated kinetic parameters | $\rho_1,\rho_2,\rho_3,\rho_5,\rho_7,\rho_9$ |

The GA implementation is custom and does **not** require MATLAB's Global Optimization Toolbox. Local refinement uses MATLAB optimization routines.

---

## Validation and robustness analyses

### 100-seed stochastic robustness

PINN and GA were evaluated over 100 independent seeds for each experimental condition. GA solutions were practically invariant across random initialization in nine of ten experiments, whereas PINN displayed greater dataset-dependent seed sensitivity.

### Time-split validation

Model estimation was restricted to observations through **21 h**, and the resulting trajectories were evaluated through the complete **30 h** experimental interval. This test evaluates short-horizon predictive behavior using genuinely withheld measurements.

### Raw-data robustness

All three inverse methods were also applied directly to the unsmoothed experimental observations. This test determines whether the principal model-based conclusions depend on the Gaussian smoothing step used in the main analysis.

### Numerical solver dependence and stiffness check

Mechanistic trajectories generated from PINN- and GA-estimated parameter sets were recomputed using Heun's method, MATLAB `ode45`, and MATLAB `ode15s`. Differences between `ode45` and `ode15s` were of the order of approximately $10^{-8}$–$10^{-6}$ over the observed states, indicating negligible solver-dependent effects over the analyzed time domain. This comparison is a numerical consistency test and is not intended as a formal proof that the system is nonstiff.

### PINN loss-weight sensitivity

The physics/data weighting coefficient $\alpha$ was varied from 0.1 to 1.0. Performance improved as the physics contribution approached unity, while $\alpha=1$ removed the data-fidelity contribution and substantially degraded agreement with experimental observations. The selected value $\alpha=0.999$ therefore strongly enforces the governing equations while retaining a nonzero experimental-data contribution.

### Extended in silico simulations

Selected mechanistic trajectories were extended beyond the 30 h observation interval to examine qualitative extrapolative behavior. These simulations are not treated as validation because experimental measurements were unavailable beyond 30 h.

---

## Main findings reproduced by this repository

| Finding | Result |
|---|---|
| Model-based in-sample fitting | LS produced the lowest model-based RSS in all 10 complete-data experiments |
| GA stochastic robustness | Practically invariant across seeds in 9 of 10 experiments |
| Direct PINN reconstruction | Lower RSS than the corresponding PINN-parameterized ODE in 6 of 10 complete-data experiments |
| PINN with raw data | Direct PINN RSS was lower than its corresponding mechanistic ODE simulation in all 10 experiments |
| Time-split validation | All mechanistic solutions retained adjusted $R^2\geq0.9750$ over the complete trajectories |
| PINN short-horizon prediction | Lowest descriptive trajectory RSS in 5 of 10 time-split experiments |
| Practical identifiability | $\rho_3$ was the principal recurrent weakly constrained parameter |
| Solver dependence | State-specific `ode45` vs `ode15s` RMSE remained approximately within $10^{-8}$–$10^{-6}$ |

These results should not be interpreted as evidence that one inverse method is universally superior. Instead, the benchmark highlights complementary properties: LS provided the strongest model-based residual fit, GA the greatest stochastic reproducibility, and the direct PINN representation the greatest trajectory-level flexibility.

---

## Software requirements

| Software | Version |
|---|---|
| MATLAB | R2025b Update 5 |
| Python | 3.10 |
| TensorFlow | 2.21.0 |
| Visual Studio Code | 1.136.0 |

MATLAB routines that use `fmincon` or classical nonlinear least-squares solvers require the **Optimization Toolbox**. The custom GA itself does not require the **Global Optimization Toolbox**.

For the Python PINN implementation, the core environment uses TensorFlow together with standard scientific-computing packages such as NumPy, pandas, SciPy, and Matplotlib. Users should verify the imports in the distributed scripts when reproducing the environment.

---

## Reproducing the manuscript results

1. Run the LS, GA, or PINN implementation from its corresponding directory to estimate the model parameters for the ten experimental conditions.
2. Use the MATLAB notebook in `Manuscript_Simulations_Tables/` to reproduce the mechanistic simulations, goodness-of-fit calculations, statistical tables, and manuscript-ready outputs.
3. Run the workflows in `Tests/` independently to reproduce the time-split, raw-data, solver-dependence, $\alpha$-sensitivity, and 100-seed robustness analyses.
4. Keep the experimental data paths and output folders consistent with the directory structure used by the scripts.

Because the PINN and GA procedures are stochastic, exact computational time depends on hardware, software versions, and stopping behavior. In the computational environment used for the manuscript, one complete seed across all ten datasets required approximately 15 minutes for either PINN or GA, while the 100-seed analyses required approximately 25 h for PINN and 12 h for GA.

---

## Interpretation notes

The direct PINN neural-network trajectory and the mechanistic ODE trajectory obtained from PINN-estimated parameters are intentionally reported separately. A low neural-network reconstruction error does not necessarily imply equally precise mechanistic parameter recovery.

Likewise, a parameter estimate reaching a prescribed bound should not automatically be interpreted as evidence that its associated mechanism is unnecessary. In this study, $\rho_3$ repeatedly approached its lower bound, but it retains a mechanistic role in separating the measured substrate-dependent contribution to biomass growth from the baseline growth term. The result is therefore interpreted primarily as evidence of limited practical information for that parameter under the available observations.

---

## Open-source and reproducibility statement

The purpose of this repository is to make the inverse-problem workflows used in the manuscript transparent and reproducible. The main LS, GA, and PINN implementations, the manuscript-level MATLAB simulation workflow, and the complete validation/robustness tests are provided so that the reported comparisons can be independently inspected, reproduced, and extended.

If you use or adapt this repository, please cite the associated manuscript.

---

## Citation

A formal citation will be added after publication.

```bibtex
@article{Valle2026MechanisticInverseProblems,
  title   = {Mechanistic Nonlinear Dynamical Modeling for Parameter Estimation:
             A Comparison of Physics Informed Neural Networks and Genetic Algorithms},
  author  = {Valle, Paul A. and Salazar, Yolocuauhtli and
             P{\'a}ez-Lerma, Jesus B. and Soto-Cruz, N. Oscar and
             Coria, Luis N. and Garc{\'i}a, Iv{\'a}n A. and
             Gonz{\'a}lez-Campos, Michell V.},
  year    = {2026},
  note    = {Manuscript under review}
}
```

---

## License

No software license is specified in this README. Before public release, add a `LICENSE` file defining the terms under which the source code and data may be reused.
