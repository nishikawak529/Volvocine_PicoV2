# ICRA Paper Outline: Predicting Collective Phase Coordination from Physical Interaction Identification

September 14, 2026

**Proposed title: Predicting Collective Phase Coordination in Modular Robots from Open-Loop Identification of Physical Interactions**

This study connects robotic experiments on rhythmic coordination through local sensory feedback with the dynamical systems theory of phase oscillators through physical interaction identification and phase reduction. A single series of open-loop pairwise experiments identifies interactions mediated by the body and environment as functions relating oscillator phases to local sensor inputs. These functions are then used to predict collective phase relations under Winfree-type control. An unaveraged model is constructed by summing pairwise interactions, and a weighted Kuramoto–Sakaguchi model is derived through a common profile approximation with a sinusoidal sender profile and first-order averaging.

Validation uses one configuration of a four-module swimming robot inspired by volvocine algae. The phase sensitivity function is fixed at $z(\phi)=\sin\phi$, and the prescribed angular frequency is fixed at $\omega_0$ for all modules. Two conditions are considered, differing in the sign of the common feedback gain. Predictions from the two models, constructed from the same identification results, are compared with experimental phase-difference time series to assess qualitative agreement in steady phase relations. Modal analysis based on singular value decomposition (SVD) is presented in the Discussion as an example of interpreting the resulting gaits in terms of collective inputs and their contributions to phase adjustment.

## 1. Introduction

Introduce the generation of periodic motion using central pattern generators (CPGs) and rhythmic coordination through local sensory feedback. Collective coordination emerges as each oscillator updates its phase in response to sensory inputs from the body and environment, without explicitly communicating oscillator phases. Present quadrupedal coordination through local force feedback and swimming coordination through local hydrodynamic force sensing as representative examples. [Owaki et al., 2013](https://doi.org/10.1098/rsif.2012.0669), [Thandiackal et al., 2021](https://doi.org/10.1126/scirobotics.abf6354)

Explain coupling through phase sensitivity and periodic inputs in the Winfree model, and introduce Winfree-type control, which applies this structure to local sensor inputs. [Winfree, 1967](https://doi.org/10.1016/0022-5193(67)90051-3), [Nishikawa et al., 2026](https://doi.org/10.1109/LRA.2026.3668705)

$$
\dot\phi_j=\omega_j+\sigma z(\phi_j)s_j(t).
$$

Here, $\phi_j$ is the controller phase, $\omega_j$ is the prescribed angular frequency, $\sigma$ is the feedback gain, $z$ is the phase sensitivity function specified in the controller, and $s_j$ is the local sensor input.

Robotic experiments using local sensory feedback have demonstrated a variety of coordinated rhythmic motions. Meanwhile, the dynamical systems theory of phase oscillators provides extensive knowledge of synchronization, phase-locked states, and their stability. However, because interactions mediated by the body and environment are unknown, experimentally observed rhythmic coordination in these robots remains insufficiently connected to theoretical analysis and control design. **The objective of this study is to construct a model that connects robotic experiments with phase oscillator theory through physical interaction identification and phase reduction.**

The specific problem is to model unknown sensory interactions and predict the collective phase relations that emerge under feedback control. Sensor inputs depend on both the sender's phase and the receiver's own phase, while constructing a detailed physical model involving fluid dynamics, passive joints, and colony motion is difficult. Describe the relationship between phase coordination, energy efficiency, and postural stability as motivation for understanding the resulting phase relations.

Review related work on identifying phase coupling functions from time series and predicting collective synchronization using experimentally derived phase models. This study focuses on **identifying sensor responses in open loop separately from the control law, and constructing collective closed-loop phase dynamics from pairwise measurements**. [Kralemann et al., 2011](https://arxiv.org/abs/1102.3064), [Tokuda et al., 2019](https://arxiv.org/abs/1904.11289), [Kiss et al., 2005](https://doi.org/10.1103/PhysRevLett.94.248301)

The paper presents three contributions:

1. **Interaction identification and construction of a collective model.** Sensor functions of two phases, identified in open loop, are separated into self and interaction terms. The collective input is modeled by summing the interaction terms.
2. **Common profile approximation and connection to a phase coupling model.** Interactions are represented using a common receiver profile, a sinusoidal sender profile, and directed coupling weights. First-order averaging yields a weighted Kuramoto–Sakaguchi model.
3. **Validation of closed-loop predictions for positive and negative gains.** Predictions from two models using the same fixed identification results are compared with experimental phase-difference time series to demonstrate qualitative agreement in the resulting gaits.

Together, these contributions provide a basis for analyzing and designing experimentally observed collective phase coordination within phase oscillator theory. Identifying an interaction model separately from the control law supports further research on optimizing the profile of the phase sensitivity function $z$ and analytically understanding the formation and stability of steady phase relations.

## 2. Problem Formulation and Robotic Platform

### 2.1 Problem Formulation

Consider a system of $N$ modules driven by local sensory feedback. Let $\boldsymbol\phi=(\phi_1,\ldots,\phi_N)^\mathsf T$ denote the phase vector. The phase of each module evolves according to

$$
\dot\phi_j=\omega_j+\sigma z(\phi_j)s_j(t)
$$

Within the operating regime considered, assume that each sensor input can be described by an unknown function $s^j$ of all module phases:

$$
\boxed{
s_j(t)=s^j\!\left(\boldsymbol\phi(t)\right),
\qquad j=1,\ldots,N
}
$$

Each $s^j$ is $2\pi$-periodic in every phase argument. **The objective of system identification is to obtain a model $\widehat s^{\,j}(\boldsymbol\phi)$ of this unknown sensory mapping $s^j(\boldsymbol\phi)$.**

Identification of the sensory mapping must be completed within a single series of open-loop experiments. The resulting model is then held fixed and used to predict collective closed-loop phase dynamics under the control conditions of interest.

The model is constructed in two stages:

1. **Modeling the sensor input.** Identify $\widehat s^{\,j}(\boldsymbol\phi)$ from open-loop measurements and substitute it into the known control law to construct an unaveraged closed-loop model.
2. **Deriving phase equations through phase reduction.** Apply a common profile approximation and phase reduction through first-order averaging to the resulting system, obtaining phase equations with coupling functions that depend on phase differences.

These two stages construct a collective model from measured sensory interactions and connect it to a weighted Kuramoto–Sakaguchi description.

### 2.2 Robotic Platform

Introduce the swimming modular robot inspired by volvocine algae, including its active joints, passive joints, flex sensors, and four-module colony structure. The commanded active-joint angle is

$$
\alpha_j^{\mathrm{act}}=A_{\mathrm{stroke}}\cos\phi_j
$$

and $s_j$ is the normalized passive-joint angle. Describe the colony configuration, module orientations and IDs, and sensor input normalization.

Validation uses one configuration of four modules. Among the control design variables $z,\sigma,\omega_j$, the following are fixed and shared by all modules:

$$
z(\phi)=\sin\phi,\qquad \omega_j=\omega_0
$$

Two conditions are considered by changing the sign of the common gain $\sigma$.

| Validation condition | Phase sensitivity function | Prescribed angular frequency | Common gain |
|---|---|---|---|
| Positive gain | $z(\phi)=\sin\phi$ | $\omega_j=\omega_0$ | $\sigma=+\sigma_0$ |
| Negative gain | $z(\phi)=\sin\phi$ | $\omega_j=\omega_0$ | $\sigma=-\sigma_0$ |

Here, $\sigma_0>0$ is the fixed gain magnitude used for validation. For both conditions, predictions from the unaveraged model and the weighted Kuramoto–Sakaguchi model, using the same identification results, are compared with experimental phase-difference time series.

## 3. Modeling: Interaction Identification and Reduction to a Kuramoto–Sakaguchi Model

### 3.1 Pairwise Identification and Construction of the Unaveraged Model

Identification covers all six pairs among the four modules. Both sensor signals are recorded for each pair, yielding 12 directed interactions.

For each pair $(j,k)$, phase feedback is disabled, and the modules are driven with a small difference between their known prescribed angular frequencies:

$$
\dot\phi_j=\omega_j^{\mathrm{OL}},\qquad
\dot\phi_k=\omega_k^{\mathrm{OL}},\qquad
\omega_j^{\mathrm{OL}}\ne\omega_k^{\mathrm{OL}}.
$$

This frequency difference sweeps the relative phase. A bivariate Fourier series is identified from the measured phase pairs and sensor responses:

$$
s_{j\leftarrow k}(\phi_j,\phi_k)
\simeq
\sum_{m,n=-M}^{M}C_{mn}^{(j\leftarrow k)}
e^{i(m\phi_j+n\phi_k)}
=
 s_{j\leftarrow k}^{\mathrm{self}}(\phi_j)
+s_{j\leftarrow k}^{\mathrm{int}}(\phi_j,\phi_k).
$$

The Fourier order is $M=10$. The $n=0$ components, which are independent of the sender phase, define the self term; the $n\ne0$ components define the interaction term. Evaluate the consistency of the self term for the same receiving module across different partners and define a representative self term $s_j^{\mathrm{self}}$.

Construct the model of the sensory mapping defined in Section 2 by adding the self term and the directed pairwise interactions:

$$
\boxed{
\widehat s^{\,j}(\boldsymbol\phi)
=s_j^{\mathrm{self}}(\phi_j)
+\sum_{k\ne j}
 s_{j\leftarrow k}^{\mathrm{int}}(\phi_j,\phi_k)
}
$$

This is an additive model of nonlinear interaction functions that depend on oscillator phases. The self term is included once for each receiving module.

Substituting the identified input into the control law gives the unaveraged model for the conditions considered in this paper:

$$
\boxed{
\dot\phi_j
=\omega_0+\sigma\sin\phi_j
\left[
 s_j^{\mathrm{self}}(\phi_j)
+\sum_{k\ne j}
 s_{j\leftarrow k}^{\mathrm{int}}(\phi_j,\phi_k)
\right]
}
$$

This model retains variations in phase velocity within each cycle and constructs collective closed-loop phase dynamics from pairwise sensory measurements.

### 3.2 Common Profile Approximation and First-Order Averaging

To obtain a Kuramoto–Sakaguchi description through averaging, restrict the sender profile to a sinusoid with a phase shift and jointly approximate the interaction terms across all pairs:

$$
\boxed{
 s_{j\leftarrow k}^{\mathrm{int}}(\phi_j,\phi_k)
\simeq W_{jk}a(\phi_j)b(\phi_k),
\qquad
b(\phi)=B\cos(\phi-\delta),
\qquad W_{jj}=0
}
$$

Here, $a$ is the common receiver profile, $b$ is the common sender profile, and $W_{jk}$ is the signed directed weight from module $k$ to module $j$. The amplitude $B$ is fixed by normalization, and $\delta$ is the common phase shift of the sender waveform. The matrix $W$ is estimated as a general real matrix.

Jointly fit $a,\delta,W$ to the identified interaction functions. Specify the normalization of the receiver profile and weights, and present the resulting common profiles, network structure, and agreement with the original interaction functions.

First-order averaging considers conditions in which feedback-induced corrections to phase velocity are small relative to the prescribed angular frequency. Let $\theta_j$ denote the averaged phase, and define the phase difference as receiver minus sender: $\psi=\theta_j-\theta_k$. The common coupling function is

$$
\Gamma_0^{(z)}(\psi)
=\left\langle z(\eta)a(\eta)b(\eta-\psi)\right\rangle
=c_z\cos\psi+d_z\sin\psi,
$$

$$
c_z=B\left\langle z(\eta)a(\eta)\cos(\eta-\delta)\right\rangle,
\qquad
d_z=B\left\langle z(\eta)a(\eta)\sin(\eta-\delta)\right\rangle
$$

where $\langle f\rangle=(2\pi)^{-1}\int_0^{2\pi}f(\eta)\,d\eta$ and $z(\eta)=\sin\eta$. Restricting the sender profile to a single sinusoid makes the coupling function a single sinusoid of the phase difference.

Define

$$
R_z=\sqrt{c_z^2+d_z^2},\qquad
\alpha_z=\operatorname{atan2}(c_z,-d_z)
$$

For $R_z>0$, the coupling function can then be written as $\Gamma_0^{(z)}(\psi)=R_z\sin(-\psi+\alpha_z)$. Averaging the self term as well and defining the effective angular frequency as

$$
\bar\omega_j
=\omega_0+\sigma
\left\langle z(\eta)s_j^{\mathrm{self}}(\eta)\right\rangle
$$

yields a weighted Kuramoto–Sakaguchi model. [Sakaguchi and Kuramoto, 1986](https://doi.org/10.1143/PTP.76.576)

$$
\boxed{
\dot\theta_j
=\bar\omega_j+
\sigma R_z\sum_{k\ne j}W_{jk}
\sin(\theta_k-\theta_j+\alpha_z)
}
$$

The effective phase shift $\alpha_z$ is calculated from the sender waveform shift $\delta$, the receiver profile $a$, and the phase sensitivity function $z$. Differences in the self terms produce module-specific effective angular frequencies $\bar\omega_j$ from the common prescribed angular frequency $\omega_0$.

The same $s^{\mathrm{self}},s^{\mathrm{int}},a,b,W,R_z,\alpha_z$ are used for both gain conditions. Predictions account for the sign of $\sigma$ and the corresponding change in effective angular frequencies.

## 4. Validation: Comparison of the Two Models with Robot Experiments

The two models compared are the unaveraged model using the pairwise interactions directly and the weighted Kuramoto–Sakaguchi model obtained through the common profile approximation and first-order averaging.

| Comparison target | Dynamics represented |
|---|---|
| Unaveraged model | Substitute $s_j^{\mathrm{self}}+\sum_{k\ne j}s_{j\leftarrow k}^{\mathrm{int}}$ into the Winfree-type control law |
| Weighted Kuramoto–Sakaguchi model | Constructed through the common profile approximation and first-order averaging, including the self terms |
| Physical robot | Four-module colony with the same $z=\sin\phi$ and $\omega_0$, under positive and negative $\sigma$ |

Both models are constructed from the same series of open-loop measurements. For each gain condition, present the experimental and simulated phase-difference time series side by side.

Let $r$ denote the reference module. Plot $\phi_j(t)-\phi_r(t)$ for the robot and the unaveraged model, and $\theta_j(t)-\theta_r(t)$ for the averaged model. Use consistent module colors, phase-difference signs, and axis ranges to compare the following qualitatively:

- The different steady phase relations produced by positive and negative gains.
- Agreement in the phase differences approached by each module.
- The collective phase patterns reproduced by the unaveraged and averaged models.

The primary evaluation concerns agreement in steady phase relations as observed in the phase-difference time series. Present the variations within each cycle seen in the robot and the unaveraged model alongside the smooth phase evolution of the averaged model.

This comparison evaluates whether collective phase relations can be predicted from open-loop pairwise measurements and whether the compact description combining the common profile approximation and averaging also reproduces the resulting gaits.

## 5. Discussion: Interpreting Steady Gaits through Modal Decomposition

Decompose the identified directed weight matrix using SVD:

$$
W=U\Sigma V^\mathsf T
=\sum_\ell s_\ell u_\ell v_\ell^\mathsf T.
$$

Here, $s_\ell$ denotes a singular value. For the unaveraged phases, define

$$
x_k=e^{i\phi_k},\qquad
Z_\ell=v_\ell^\mathsf T\boldsymbol x,\qquad
G_j=(W\boldsymbol x)_j
=\sum_\ell s_\ell u_{j\ell}Z_\ell
$$

The interaction component of the sensory input under the common profile approximation is then

$$
\widehat s_j^{\mathrm{int}}
=B\,a(\phi_j)\operatorname{Re}\left[e^{-i\delta}G_j\right]
$$

The quantity $Z_\ell$ is the collective sender signal for each mode, and $s_\ell u_{j\ell}$ is the coefficient determining that signal's contribution to the receiver. This decomposition represents the collective input received by each module as a superposition of interaction modes.

For the averaged phases, similarly define $Z_\ell(\boldsymbol\theta)=v_\ell^\mathsf T e^{i\boldsymbol\theta}$. The contribution of each mode to phase velocity is

$$
f_{j\ell}
=\sigma R_zs_\ell u_{j\ell}
\operatorname{Im}\left[e^{i(\alpha_z-\theta_j)}Z_\ell(\boldsymbol\theta)\right]
$$

In a phase-locked state, all modules satisfy

$$
\bar\omega_j+\sum_\ell f_{j\ell}=\Omega
$$

where $\Omega$ is the common angular frequency.

For the two gaits obtained with positive and negative gains, compare the amplitudes and relative phases of the collective signals and their contributions to the receivers. Interpret each steady state in terms of which modes adjust each oscillator's phase velocity and how these contributions establish a common collective frequency.

Position this modeling approach as a method for connecting local sensory interactions with phase oscillator theory. Discuss its range of applicability in terms of the assumptions underlying the sensory mapping as a function of phases, the addition of pairwise interactions, the common profile approximation, and first-order averaging. Describe extensions to phase sensitivity function design and control of desired collective phase relations.

## 6. Conclusion

A single series of open-loop pairwise experiments provides a sensor input model separated into self and interaction terms. A common profile approximation with a sinusoidal sender profile, followed by first-order averaging, expresses the identified interactions as a weighted Kuramoto–Sakaguchi model.

With the common $z(\phi)=\sin\phi$ and prescribed angular frequency fixed, phase-difference time series from the unaveraged model, the weighted Kuramoto–Sakaguchi model, and the physical robot are compared under positive and negative gains. Summarize the qualitative agreement in steady phase relations and the significance of predicting collective phase coordination under different feedback conditions using the same identified sensory interactions.

The central significance of this study is that a model constructed from identification connects robotic experiments on rhythmic coordination through local sensory feedback with the dynamical systems theory of phase oscillators. Constructing a phase coupling model from measured physical interactions and demonstrating agreement between its predictions and robot experiments connects experimentally observed coordinated motion with theoretical analysis and control design.

The interpretation of steady gaits through SVD-based modal analysis provides one example of this connection. The resulting model provides a basis for further research on optimizing the profile of the phase sensitivity function $z$ and analytically understanding the formation and stability of steady phase relations.

## Key References

- Winfree (1967), *Biological rhythms and the behavior of populations of coupled oscillators*. [Paper](https://doi.org/10.1016/0022-5193(67)90051-3)
- Nishikawa, Dan and Kurabayashi (2026), *Winfree-Model-Type Synchronization Control for Swimming Modular Robot Through Physical Interaction*. [Paper](https://doi.org/10.1109/LRA.2026.3668705)
- Owaki et al. (2013), *Simple robot suggests physical interlimb communication is essential for quadruped walking*. [Paper](https://doi.org/10.1098/rsif.2012.0669)
- Thandiackal et al. (2021), *Emergence of robust self-organized undulatory swimming based on local hydrodynamic force sensing*. [Paper](https://doi.org/10.1126/scirobotics.abf6354)
- Kralemann, Pikovsky and Rosenblum (2011), *Reconstructing phase dynamics of oscillator networks*. [Author manuscript](https://arxiv.org/abs/1102.3064)
- Tokuda, Levnajić and Ishimura (2019), *A practical method for estimating coupling functions in complex dynamical systems*. [Author manuscript](https://arxiv.org/abs/1904.11289)
- Kiss, Zhai and Hudson (2005), *Predicting Mutual Entrainment of Oscillators with Experiment-Based Phase Models*. [Paper](https://doi.org/10.1103/PhysRevLett.94.248301)
- Sakaguchi and Kuramoto (1986), *A Soluble Active Rotator Model Showing Phase Transitions via Mutual Entrainment*. [Paper](https://doi.org/10.1143/PTP.76.576)
