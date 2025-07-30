---
title: 
author: 
pages: 12
conversion_method: pymupdf4llm
converted_at: 2025-07-30T20:07:58.467646
---

# 

## Bath energy for correlated initial states versus information flow in local dephasing channels

Filippo Giraldi


School of Chemistry and Physics, University of KwaZulu-Natal
and National Institute for Theoretical Physics (NITheP)
Westville Campus, Durban 4000, South Africa


Gruppo Nazionale per la Fisica Matematica (GNFM-INdAM)
c/o Istituto Nazionale di Alta Matematica Francesco Severi
Cittá Universitaria, Piazza Aldo Moro 5, 00185 Roma, Italy


PACS: 03.65.Yz, 03.65.Ta


**Abstract**


Variations of the bath energy are compared with the information flow in local dephasing channels. Special correlated initial conditions are prepared from the thermal
equilibrium of the whole system, by performing a selective measurement on the qubit.
The spectral densities under study are ohmic-like at low frequencies and include logarithmic perturbations of the power-law profiles. The bath and the correlation energy
alternately increase or decrease, monotonically, over long times, according to the value of
the ohmicity parameter, following logarithmic and power laws. Consider initial conditions
such that the environment is in a thermal state, factorized from the state of the qubit.
In the super-ohmic regime the long-time features of the information flow are transferred
to the bath and correlation energy, by changing the initial condition from the factorized
to the specially correlated, even with different temperatures. In fact, the low-frequency
structures of the spectral density that provide information backflow with the factorized
initial condition, induce increasing (decreasing) bath (correlation) energy with the specially correlated initial configuration. By performing the same change of initial conditions,
the spectral properties providing information loss, produce decrease (increase) of the bath
(correlation) energy.

### **1 Introduction**


Nowadays, increasing interest has been devoted to the study of the exchange of energy between
an open quantum system and the external environment and to the connections with the flow
of quantum information [1, 2, 3]. The analysis performed in Ref. [1] has shown that backflow
of energy can be observed in regime of non-Markovianity. Such regime can be interpreted as a
flow of quantum information from the environment back in the open system [4, 5, 6, 7, 8]. In
local dephasing channels the qubit experiences pure dephasing and no dissipation of energy
and the appearance of information backflow is related to the structure of the environmental
spectrum [7, 8, 9]. In fact, for spectral densities (SDs) that are ohmic-like at low frequencies,
information backflow appears uniquely if the ohmicity parameter is larger than a critical value
which depends on the temperature of the thermal bath [7].
In order to find further connections between energy and information flow, here, we consider
a qubit that interacts locally with a thermal bath and experiences pure dephasing and no
dissipation of energy, for factorized [7] or special correlated initial conditions [10, 11, 12].
By focusing on the correlated initial configurations, we study how the short- and long-time


1


behaviors of the bath or correlation energy depend on the environmental spectrum. The SDs
under study are ohmic-like at low frequencies and include possible logarithmic perturbations
of the power-law profiles [13, 9]. For factorized initial states of the qubit and the thermal
bath, the directions of the long-time information flow, back in the open system or forth in
the external environment, exhibit regular patterns that dependent on the ohmicity parameter

[9]. In light of these results, we search for connections among variations of the bath or
correlation energy, the directions of the information flow and the environmental spectrum in
local dephasing channels, for the factorized and the special correlated initial conditions.
The paper is organized as follows. Section 2 is devoted to the description of the model and
the initial conditions. The general class of ohmic-like SDs with logarithmic perturbations is
defined in Section 3 The bath and correlation energy and the short- or long-time behavior are
analyzed in Section 4 The variations of the bath and correlation energy are compared with
the information flow in Section 5 A summary of the result is provided along with conclusions
in Section 6 Details of the calculations are given in the Appendix.

### **2 Model and initial conditions**


The system under study consists in a qubit that interacts locally with a reservoir of field
modes [7, 14, 15] according to the microscopic Hamiltonian H, where H = H S + H SE + H E .
The Hamiltonian of the system, H S, the Hamiltonian of the external environment, H E, and
the interaction Hamiltonian, H SE, are given by



σ z �g k b k + g k [∗] [b] [†] k �, H E = �
k k



H S = ω 0 σ z, H SE = �



ω k b [†] k [b] [k] [.] (1)

k



In the chosen system of units the Planck and Boltzmann constants are equal to unity, ℏ =
k B = 1. The transition frequency of the qubit is represented by the parameter ω 0 . The rising
and lowering operator of the kth mode are b [†] k [and][ b] [k] [, respectively, while][ ω] [k] [ represents the]
frequency of the same mode. The coefficient g k is the coupling strength between the qubit and
the kth frequency mode. The index k runs over the frequency modes. The z-component of
the Pauli spin operator [16, 17] is referred as σ z . The mixed state of the qubit is described by
the reduced density matrix ρ(t), at the time t, and is obtained by tracing the density matrix
of the whole system, at the time t, over the Hilbert space of the external environment [16].
The model is exactly solvable [18, 19, 20] and describes a pure dephasing process of the qubit.
Consider initial conditions such that the qubit is decoupled from the external environment,
which is represented by a structured reservoir of field modes or by a thermal bath. In the
interaction picture, the reduced density matrix describing the qubit, evolves according to the
master equation

˙
ρ(t) = γ(t) (σ z ρ(t)σ z − ρ(t)) . (2)


The function γ(t) represents the dephasing rate and depends on the SD of the system. At
zero temperature, T = 0, the dephasing rate is labeled here as γ 0 (t) and reads



∞
γ 0 (t) =
ˆ 0



J (ω)

sin (ωt) dω. (3)
ω



The function J (ω) represents the SD of the system and is defined in terms of the coupling
constants g k via the following form,


J (ω) = � |g k | [2] δ (ω − ω k ) . (4)


k


2


If the external environment is initially in a thermal state, T > 0, the dephasing rate is
represented here as γ T (t) and reads



∞
γ T (t) =
ˆ 0



J T (ω)

dω, (5)
ω



where the effective SD J T (ω) is defined for every non-vanishing temperature T as


J T (ω) = J (ω) coth [ω] (6)

2T [.]


For the system under study, the trace distance measure of non-Markovianity [4] provides a
simple expression of the non-Markovianity measure [5, 6, 7, 8],


N = |γ(t)| e [−][Ξ(][t][)] dt. (7)
ˆ γ(t)<0


The open dynamics is Markovian if the dephasing rate is non-negative. On the contrary,
persistent negative values of the dephasing rate witness non-Markovianity and are interpreted
as a flow of quantum information from the environment back in the system. Refer to [5, 6, 7, 8]
for details.

The energy of the open system can be interpreted as the expectation value of the Hamiltonian H S and is constant, since the Hamiltonian H S commutes with the total Hamiltonian
H. The populations of the excited and ground state of the qubit are also constant. The nonequilibrium energy of the bath and the correlation energy can be estimated as the expectation
values of the bath Hamiltonian, H E, and the interaction Hamiltonian, H SE, respectively. The
sum of the two energies is referred as the environmental energy and is constant. See Refs.

[10, 11, 12] for details.


**2.1** **Special correlated initial states**


The analysis performed in Ref. [11] has shown how the reduced density matrix of the bath, the
non-equilibrium energy of the bath, named there as phonon energy, and the non-equilibrium
correlation energy evolve in time for special correlated initial states. For the sake of clarity,
the main expressions describing such initial conditions, the non-equilibrium energy of the bath
and the correlation energy are reported below.
Following Refs. [10, 11, 12], the special correlated initial conditions are prepared from the
thermal equilibrium of the whole system at temperature T . The system is described by the

−
density matrix exp ( H/T ) /Z 0 [′] [, where][ Z] 0 [′] [is a normalization constant,][ Z] 0 [′] [= Tr][ {][exp (][−][H/T] [)][}][.]
The symbol Tr denotes the trace operation over the Hilbert space of the whole system.
The qubit is prepared in a pure state |φ 0 ⟩ via a selective measurement [21, 22] that induces the whole system in the state P 0 exp (−H/T ) P 0 /Z 0 . The projector operator P 0 represents the effect of the measurement, P 0 = |φ 0 ⟩⟨φ 0 |, while Z 0 is a normalization constant,
Z 0 = Tr {P 0 exp (−H/T ) P 0 }. In this way, the whole system is prepared in the initial condition ρ(0), given by ρ(0) = |φ 0 ⟩⟨φ 0 | ⊗ ρ E (0). The mixed state of the external environment,
ρ E (0), is given by


⟨φ 0 |exp (−H/T )| φ 0 ⟩
ρ E (0) = Tr E ⟨φ 0 |exp (−H/T )| φ 0 ⟩ [.] (8)


The symbol Tr E denotes the trace operation over the Hilbert space of the external environment. The thermal equilibrium of the whole system and the selective measurement on the
qubit constrain the thermal bath in the mixed state ρ E (0). Such state depends on the state
|φ 0 ⟩ of the qubit and on the interaction Hamiltonian H SE . Consequently, in such initial configuration the qubit and thermal bath are correlated. Refer to [10, 11, 12, 23, 24, 25, 26, 27]
for details.


3


### **3 Ohmic-like spectral densities and logarithmic perturbations**



The SDs that are usually considered in literature [23, 17] are ohmic-like at low frequencies
and exhibit an exponential cut-off at high frequencies, J (ω) ∝ ω c (ω/ω c ) [α] [0] exp (−ω/ω c ), are
Lorentzian or are characterized by a finite support, J (ω) ∝ ω c (ω/ω c ) [α] [0] for every ω ∈ [0, ω c ],
where ω c < ∞. The power α 0 is referred as the ohmicity parameter, while ω c is labeled
the cut-off frequency of the reservoir spectrum. The ohmic-like SDs are named sub-ohmic if
0 < α 0 < 1, ohmic if α 0 = 1 and super-ohmic if α 0 - 1. The description of an experimental
setting is beyond the purposes of the present paper. Still, it is worth mentioning that ohmiclike SDs can be engineered in cold environments [28, 29]. An impurity that is embedded in
a double well potential and is immersed in a cold gas reproduces, under suitable conditions,
a qubit that interacts with an ohmic-like environment. The ohmicity parameter changes by
varying the scattering length and the dimension of the gas, one-, two- or three-dimensional.
See Refs. [28, 29] for details.
Coherence between the two states of a qubit, backflow of quantum information and the
non-equilibrium energy of the reservoir have been largely investigated for ohmic-like SDs

[17, 16, 18, 19, 20, 7, 8, 10, 11, 12]. A review of the results is beyond the purposes of the
present paper. In light of these studies, one might wonder how the non-Markovian dynamics,
the coherence and the non-equilibrium energy of the environment are affected if the SD of
the system slightly departs from the physically feasible power-law profiles of the ohmic-like
condition at low frequencies. For this reason, two general classes of SDs have been recently
considered. In each class, the SDs include the ohmic-like condition, as a particular case, and
depart from the low-frequency power laws according to natural or arbitrarily real powers of
logarithmic forms [13]. The long-time decoherence or recoherence process and the backflow of
information are uniquely determined by the ohmicity parameter and, except for special cases,
are not affected by the mentioned logarithmic perturbations of the power-law profiles of the
ohmic-like SDs [9]. In light of these results, we intend to analyze the short- and long-time
behavior of the bath energy by considering the ohmic-like and logarithmic-like SDs that are
introduced in Refs. [13]. For the sake of clarity, we report the definitions and the involved
constraints below. For continuous distributions of frequency modes the following constraint
on the SD holds [30, 31],

∞

J (ω)

dω < ∞. (9)

ˆ 0 ω



J (ω)



0



dω < ∞. (9)
ω



The SDs under study are described via the dimensionless auxiliary function Ω(ν) which is
defined via the scaling property Ω(ν) = J (ω s ν) /ω s, where ω s is a typical scale frequency of
the system.


**3.1** **First class of spectral densities**


The first class of SDs is defined by auxiliary functions that are continuous for every ν > 0
and exhibit as ν → 0 [+] the asymptotic behavior [32]



n j
� c j,k ν [α] [j] (− ln ν) [k], (10)

k=0



Ω(ν) ∼



∞
�

j=0



where α 0 - 0, ∞ - n j ≥ 0, α j+1 - α j for every j ≥ 0, and α j ↑ +∞ as j → +∞. Since the
auxiliary functions Ω(ν) are non-negative, the coefficients c j,k must be chosen accordingly,
and the constraint c 0,n 0 - 0 is required. The particular case n 0 = 0 provides ohmic-like
SDs with ohmicity parameter α 0 . Consequently, at low frequencies, ω ≪ ω s, for n 0 = 0 the
corresponding SDs are super-ohmic for α 0 - 1, ohmic for α 0 = 1 and sub-ohmic for 0 < α 0 < 1


4


[17, 23]. The logarithmic singularity in ν = 0 is removed by defining Ω(0) = 0. The above
properties and the asymptotic behavior Ω(ν) = O �ν [−][1][−][χ] [0] [�] as ν → +∞, where χ 0 - 0,
guarantee the required summability of the SDs. Notice that the fundamental constraint (9) is
fulfilled since the SDs are continuous, α 0 - 0 and χ 0 - 0. Additionally, the Mellin transform
ˆΩ(s) of the auxiliary functions Ω(ν) and the meromorphic continuation [32, 33] are required to
decay sufficiently fast as |Im s| → +∞. Details are provided in the Appendix and in Ref. [13].
The definition of this class of SDs is quite simple but the logarithmic powers are restricted to
natural values. Arbitrarily positive or negative, or vanishing powers of logarithmic forms are
considered in the second class of SDs which is defined below. The insertion of this arbitrariness
requires more constraints but allows to perturb the power laws of the ohmic-like profiles with
arbitrarily small, positive or negative, powers of logarithmic functions. In fact, arbitrarily
small, positive (negative) values of the first logarithmic power β 0 provide arbitrarily small
increases (decreases) in the power-low profiles. In this way, one can evaluate the accuracy of
the results that are obtained for the experimentally feasible ohmic-like SDs with respect to
logarithmic variations of the low-frequency power-law profiles.


**3.2** **Second class of spectral densities**


The second class of SDs is described by auxiliary functions that exhibit as ν → 0 [+] the
asymptotic expansion



Ω(ν) ∼



∞
� w j ν [α] [j] (− ln ν) [β] [j] . (11)

j=0



The powers α j fulfill the constraints mentioned in Sec. 3.1, while the powers β j are arbitrarily
real, either positive or negative, or vanishing. Since the auxiliary functions Ω(ν) are nonnegative, the coefficients w j must be chosen accordingly, and the constraint w 0 - 0 is required.
Again, the logarithmic singularity in ν = 0 is removed by setting Ω(0) = 0. The auxiliary
functions Ω(ν) are required to be continuous and differentiable in the support and summable.
Again, the summability condition and the choice α 0 - 0 guarantee that the fundamental
constraint (9) holds. Let ¯n be the least natural number such that ¯n ≥ α k¯, where α k¯ is
the least of the powers α k that are larger than or equal to unity. The function Ω [(¯][n][)] (ν)
is defined as the ¯nth derivative of the auxiliary function and is required to be continuous
on the interval (0, ∞). The integral ´ ∞0 [Ω(][ν][) exp (][−][ıξν][)][ dν][ has to converge uniformly for]
all sufficiently large values of the variable ξ and the integral ´ Ω [(¯][n][)] (ν) exp (−ıξν) dν must
converge at ν = +∞ uniformly for all sufficiently large values of the variable ξ. The auxiliary
functions are required to be differentiable k times and the corresponding derivatives must
fulfill as ν → 0 [+] the asymptotic expansion



dν [k]



Ω [(][k][)] (ν) ∼



∞
�



∞ d [k]
� w j dν

j=0



�ν [α] [j] (− ln ν) [β] [j] [�],



for every k = 0, 1, . . ., ¯n, where ¯n is the non-vanishing natural number defined above. Furthermore, for every k = 0, . . ., ¯n − 1, the function Ω [(][k][)] (ν) must vanish in the limit ν → +∞. The
above constraints are based on the asymptotic analysis performed in Ref. [34]. Notice that in
both the classes of SDs under study the auxiliary functions Ω(ν) are non-negative, bounded
and summable, due to physical grounds, and, apart from the above constraints, arbitrarily
tailored.


5


### **4 The bath energy**

Following Ref. [11], the non-equilibrium energy of the bath, ǫ E (t), is evaluated as


ǫ E (t) = � ω k n k (t), (12)


k


where n k (t) = Tr ρ(0)b [†] k [(][t][)][b] [k] [(][t][)] . The operators b [†] k [(][t][)][ and][ b] [k] [(][t][)][ represent the rising and]
� �
lowering operators of the kth frequency mode in the Heisemberg picture, respectively, at time
t. The index k runs over the frequency modes. Let the whole system be initially set in the
special correlated condition ρ(0), given by ρ(0) = |φ 0 ⟩⟨φ 0 | ⊗ ρ E (0) and described in Sec. 2.1
The initial mixed state of the environment, ρ E (0), is given by Eq. (8). Under such initial
condition, the non-equilibrium energy of the bath [11] is given by


ǫ E (t) = ǫ E (0) + d 0 (η 1 − Λ(t)) . (13)


If the distribution of frequency modes is discrete, the initial bath energy [11] is given by



ǫ E (0) = �


k



ω k (14)
exp (ω k /T ) − 1 [+][ η] [1] [,]



while, for a continuous distribution of frequency modes, the initial bath energy reads



∞
ǫ E (0) =
ˆ 0



ωr (ω)
(15)
exp (ω/T ) − 1 [dω][ +][ η] [1] [.]



The function r (ω) denotes the density of the modes [16, 17] at frequency ω. The parameter
η 1 is the first negative moment of the SD,



∞

η 1 =
ˆ 0



J (ω)

dω,
ω



while the parameter d 0 is defined [11] as


d 0 = 2 1 + ⟨ φ 0 |σ 3 | φ 0 ⟩ [sinh (][ω] [0] [/T] [)][ −⟨] [φ] [0] [ |][σ] [3] [|][ φ] [0] [⟩] [cosh (][ω] [0] [/T] [)]
� cosh (ω 0 /T ) −⟨ φ 0 |σ 3 | φ 0 ⟩ sinh (ω 0 /T )



.
�



The parameter d 0 vanishes for |⟨ φ 0 |σ 3 | φ 0 ⟩| = 1, and is positive for
|⟨ φ 0 |σ 3 | φ 0 ⟩| < 1. Consequently, the bath energy ǫ E (t) is constant, ǫ E (t) = ǫ E (0), for
|⟨ φ 0 |σ 3 | φ 0 ⟩| = 1, while it is time-dependent for |⟨ φ 0 |σ 3 | φ 0 ⟩| < 1. The function Λ(t) reads

[11]



∞
Λ(t) =
ˆ 0



J (ω)

cos (ωt) dω, (16)
ω



and drives the evolution of the bath energy via Eq. (13). This function can be studied in
terms of the SD of the system by following the analysis of the bath correlation function that
is performed in Ref. [13].


**4.1** **Short- and long-time behavior**


Following Ref. [11], the interaction energy ǫ SE (t) is defined as the expectation value of the
interaction Hamiltonian H SE and reads


ǫ SE (t) = Tr �ρ(0)e [ıHt] H SE e [−][ıHt] [�] . (17)


6


The expectation value of the term (H SE + H E ) of the Hamiltonian is constant and is referred
as the environmental energy ǫ env . Such energy is given by ǫ env = Tr {ρ(0) (H SE + H E )}. Since
the environmental energy is constant, the correlation energy can be evaluated from the bath
energy as ǫ SE (t) = ǫ env − ǫ E (t).
At this stage, we start our analysis. We study the short- and long-time behavior of the
bath and correlation energy. Over short times the bath energy depends on integral and high
frequency properties of the SD. If the SD belongs to the first class (Sec. 3.1) and decays
sufficiently fast over high frequencies, χ 0 - 1, the bath energy increases quadratically in time
for t ≪ 1/ω s,


ǫ E (t) ∼ ǫ E (0) + l E t [2], (18)


where l E = d 0 ´ ∞0 ωJ (ω) dω/2. If the SD belongs to the second class (Sec. 3.2) and χ 0 - 3,
the short-time behavior of the bath energy is the same as the one found for the first class,
Eq. (18). Over long times, t ≫ 1/ω s, the bath energy tends to the asymptotic value ǫ E (∞),
given by


ǫ E (∞) = ǫ E (0) + d 0 η 1, (19)


while the correlation energy tends to the asymptotic value ǫ SE (∞), given by ǫ SE (∞) =
ǫ env − ǫ E (∞). If the ohmicity parameter is not an odd natural number the bath energy
exhibits logarithmic relaxations for t ≫ 1/ω s,


ǫ E (t) ∼ ǫ E (∞) + u 0 (ω s t) [−][α] [0] ln [n] [0] (ω s t), (20)


where u 0 = −ω s d 0 c 0,n 0 cos (πα 0 /2) Γ (α 0 ). The above relaxations turn into dominant inverse
power laws for n 0 = 0,


ǫ E (t) ∼ ǫ E (∞) + u [′] 0 [(][ω] [s] [t][)] [−][α] [0] [,] (21)


where u [′] 0 [=][ −][ω] [s] [d] [0] [c] [0][,][0] [ cos (][πα] [0] [/][2) Γ (][α] [0] [)][. If the ohmicity parameter is an odd natural number,]
α 0 = 1 + 2m 0, where m 0 is a natural number, and n 0 is a non-vanishing natural number, the
bath energy relaxes over long times, t ≫ 1/ω s, as


ǫ E (t) ∼ ǫ E (∞) + u 1 (ω s t) [−][1][−][2][m] [0] ln [n] [0] [−][1] (ω s t), (22)


where u 1 = (−1) [1+][m] [0] πn 0 (2m 0 )!ω s d 0 c 0,n 0 /2. The above relaxations become dominant inverse
power laws if n 0 = 1,


ǫ E (t) ∼ ǫ E (∞) + u [′] 1 [(][ω] [s] [t][)] [−][1][−][2][m] [0] [,] (23)


where u [′] 1 [= (][−][1)] [1+][m] [0] [π][ (2][m] [0] [)!][ω] [s] [d] [0] [c] [0][,][1] [/][2][. Faster relaxations of the bath energy to the asymp-]
totic value ǫ E (∞) appear if the ohmicity parameter takes odd natural values and if n 0 vanishes. Let k 0 be the least non-vanishing index such that α k 0 is not an odd natural number,
or α k 0 = 1 + 2m k 0, where m k 0 and n k 0 are non-vanishing natural numbers. We consider SDs
such that the index k 0 exists with the required properties. The long-time behavior of the bath
energy is obtained, in the former case, from Eqs. (20) and (21) by substituting the parameter
α 0 with α k 0 and n 0 with n k 0, and, in the latter case, from Eqs. (22) and (23) by substituting
the parameter m 0 with m k 0 and n 0 with n k 0 .
For the second class of SDs, the bath energy tends over long times, t ≫ 1/ω s, to the
asymptotic value ǫ E (∞) with relaxations that involve arbitrarily positive or negative, or
vanishing powers of logarithmic forms,


ǫ E (t) ∼ ǫ E (∞) + (ω s t) [−][α] [0] [ �] u 2 ln [β] [0] (ω s t) + u [′] 2 [ln] [β] [0] [−][1] [ (][ω] [s] [t][)], (24)
�


7


where u [′] 2 [=][ w] [0] [d] [0] [ω] [s] [β] [0] �cos (πα 0 /2) Γ [(1)] (α 0 ) − π sin (πα 0 /2) Γ (α 0 ) /2� and u 2 = w 0 u 0 /c 0,n 0 .
If the ohmicity parameter α 0 differs from odd natural values, the dominant part of the above
asymptotic form is


ǫ E (t) ∼ ǫ E (∞) + u 2 (ω s t) [−][α] [0] ln [β] [0] (ω s t), (25)


and turns into power laws for β 0 = 0,


ǫ E (t) ∼ ǫ E (∞) + u 2 (ω s t) [−][α] [0] . (26)


If the ohmicity parameter is an odd natural number and β 0 does not vanish, the bath energy
tends to the asymptotic value as


ǫ E (t) ∼ ǫ E (∞) + u [′] 2 [(][ω] [s] [t][)] [−][α] [0] [ ln] [β] [0] [−][1] [ (][ω] [s] [t][)][ .] (27)


The relaxations become inverse power laws for β 0 = 1,


ǫ E (t) ∼ ǫ E (∞) + u [′] 2 [(][ω] [s] [t][)] [−][α] [0] [ .] (28)


For the second classes of SDs, the long-time relaxations of the bath energy to the asymptotic
value exhibit the same dependence on the low-frequency structure as those found for the first
class.

### **5 Variations of the bath energy versus information flow**


According to the analysis performed in the previous Section, the bath (correlation) energy
increases (decreases) quadratically over short times, t ≪ 1/ω s, if the SDs decay sufficiently
fast over high frequencies, χ 0 - 1 for the first class, or χ 0 - 3 for the second class. Over
long times, t ≫ 1/ω s, the bath and the correlation energy alternately increase or decrease,
monotonically, to the corresponding asymptotic value. The appearance of each of the two
regimes depends uniquely on the ohmicity parameter α 0 and is independent of the logarithmic
factors that possibly tailor the low-frequency structure of the SD, except for special cases
that involve odd natural values of the ohmicity parameter. In fact, for t ≫ 1/ω s, the bath
(correlation) energy increases up (decreases down) to the asymptotic value for 0 < α 0 < 1
and 3 + 4n < α 0 < 5 + 4n, where n = 0, 1, 2, . . ., if the logarithmic power n 0 does not vanish.
Same long-time increasing (decreasing) behavior appears for every odd natural value of the
ohmicity parameter if n 0 vanishes and 3+4n < α k 0 ≤ 5+4n, where n = 0, 1, 2, . . .. The power
α k 0 is defined in Sec. 4.1 Additionally, increase (decrease) of the bath (correlation) energy is
obtained for t ≫ 1/ω s if n 0 does not vanish and α 0 = 1 + 4l 0, where l 0 is natural valued. If
the ohmicity parameter differs from the values reported above, the bath (correlation) energy
decreases (increases) monotonically for t ≫ 1/ω s down (up) to the asymptotic value. For the
second class of SDs the bath and correlation energy exhibit the same long-time behavior and
the same dependence on the low-frequency profile of the SD as those obtained for the first
class. Due to the relationship with the low-frequency environmental spectrum, the loss or
gain of bath and correlation energy might be controlled and manipulated over long times by
preparing the system in the mentioned correlated initial states and by engineering ohmic-like

environments.
For the sake of clarity, we remind how the short- and long-time flow of information in
local dephasing channels depend on the low- and high-frequency structure of the SD [9]. Let
the environment be initially in a thermal state and be factorized from the initial state of the
qubit. For the first class of SDs, the information is lost in the external environment over
short times, t ≪ 1/ω s . Same short-time behavior is obtained for the second class of SDs
and χ 0 - 2. Over long times, t ≫ 1/ω s, the direction of the information flow is determined


8


by the low-frequency structure of the SD via the ohmicity parameter α 0 . At non-vanishing
temperatures, the information flows back in the open system over long times, t ≫ 1/ω s, for
3 + 4n < α 0 < 5 + 4n, where n = 0, 1, 2, . . ., if the natural logarithmic power n 0 does not
vanish, n 0 - 0. Information backflow appears also for every odd natural value of the ohmicity
parameter that differs from unity, if 3+ 4n < α k 0 ≤ 5+ 4n, where n takes natural values, and
the natural logarithmic power n 0 vanishes. Additionally, information backflow is obtained for
every odd natural value α 0 = 1 + 4l 1, where l 1 is a positive natural number, if n 0 - 0. Same
long-time behavior and same relationships with the low-frequency structure of the SD are
found in the super-ohmic regime, α 0 - 1, for the second class of SDs. Refer to [9] for details.
Straight similarities appear by comparing the present analysis of the bath and correlation
energy with the behavior of the information flow for the variety of SDs under study. If the
whole system is initially prepared in the special correlated states |φ 0 ⟩⟨φ 0 | ⊗ ρ E (0) that are
introduced in Sec. 2.1, the variations of bath and correlation energy follow the directions of the
information flow that are obtained for factorized initial conditions of the qubit and the thermal
state of the bath, even if the temperatures of the initial configurations are different. In fact,
over short times, if the SDs decay sufficiently fast at high frequencies, the increase (decrease)
of the bath (correlation) energy overlaps with the loss of information. Over long times, in
the sub-ohmic and ohmic regime, 0 < α 0 ≤ 1, for the first class of SDs, the information is
lost in the environment and the bath (correlation) energy increases (decreases) for both the
classes of SDs. Again, the temperatures in the factorized and the special correlated initial
conditions can be different. In the super-ohmic regime, α 0 - 1, perfect accordance is found,
for both the classes of SDs, between the backflow of information, obtained for the factorized
initial states, and the increase (decrease) of bath (correlation) energy occurring for the special
correlated initial conditions. Same relation appears between the decrease (increase) of bath
(correlation) energy and the loss of information in the external environment, over long times
and in the super-ohmic regime. Again, such correspondence holds even if temperatures of the
initial conditions are different. The above connections hold for logarithmic perturbations of
the low-frequency ohmic-like profiles of the SDs.

### **6 Summary and conclusions**


We have considered a qubit that interacts locally with a bosonic bath. Due to the nature
of the interaction, the qubit experiences pure dephasing and no dissipation of energy and
reproduces a local dephasing channel. The bath and the correlation energy are evaluated
via the expectation value of the bath and the interaction Hamiltonian, respectively. The
sum of the bath and the correlation energy is interpreted as the environmental energy and is
constant [11]. The whole system is initially prepared in special correlated states of the qubit
and the external environment. Such states are obtained from the thermal equilibrium of the
whole system by performing a selective measure on the qubit [10, 11, 12, 23, 24, 25, 26, 27].
We have found that, over short times, the bath (correlation) energy increases (decreases)
quadratically in time if the SDs decay sufficiently fast at high frequencies. Over long times,
the bath (correlation) energy evolves monotonically towards the asymptotic value: it increases
(decreases) in the sub-ohmic regime and also for regular intervals of the ohmicity parameter
in the super-ohmic regime, and decreases (increases) otherwise. These asymptotic behaviors
are not altered by logarithmic perturbations of the low-frequency ohmic-like profiles, except
for special conditions that involve odd natural values of the ohmicity parameter.
If the external environment is set in a thermal state and is factorized from the initial
state of the qubit, the long-time flow of quantum information exhibits regular patterns that
depend on the ohmicity parameter [9]. By changing the initial condition from the factorized
states to the special correlated states, even at different temperatures, the long-time features
of the information flow are transferred to the bath and correlation energy, in the super-ohmic


9


regime. In fact, over long times, for the variety of SDs under study and in the super-ohmic
regime, the same low-frequency spectral properties that provide backflow of information for
the factorized initial conditions, induce increase (decrease) of bath (correlation) energy if the
whole system is initially set in the special correlated states. Similarly, the low-frequency
spectral properties that provide long-time loss of information for the factorized initial states,
induce decrease (increase) of the bath (correlation) energy, if the system is initially prepared
in the special correlated states. These connections hold for ohmic-like environments and are
not altered by logarithmic perturbations of the low-frequency ohmic-like profiles of the SDs.
Even if no energy flows in the open system, the present analysis shows a straight relation
between the information flow and the variations of the bath or correlation energy, over long
times, in local dephasing channels, by properly changing the initial conditions. We believe
that the present approach is of interest in the context of control and manipulation of the bath
energy by engineering the external environment and setting the system in special correlated
initial conditions.

### **A Details**


The short- and long-time behavior of the function Λ(t), given by Eq. (16), is analyzed by
introducing the function K (τ ). Such function is defined via the scaling K (τ ) = Λ (τ/ω s ) /ω s,
and results in the form



∞
K (τ ) =
ˆ 0



Ω(ν)

cos (ντ ) dν. (29)
ν



The Mellin transform of the function K (τ ) is


ˆ π
K (s) = cos Γ(s)Ω( [ˆ] −s). (30)
� 2 [s] �


The fundamental strip is 0 < Re s < min {1, α 0 }. If χ 0 - 1 the function K [ˆ] (s) decays [35] in
the strip max {−4, −1 − χ 0 } < Re s < −2 as K [ˆ] (s) = o |Im s| [−][5][/][2] [�] for | Im s| → +∞. Such
�
decay is sufficiently fast and the singularity in s = −2 provides the asymptotic expansion (18).
Same short-time behavior is obtained for the second class of SDs if χ 0 - 3, by considering the
definition of the function Λ(t) and performing the time series expansion of the corresponding
integrand function.
As far as the long-time behavior of the function Λ(t) is concerned, let the strip µ 0 ≤
Re s ≤ δ 0 exist such that the function Ω( [ˆ] −s), or the meromorphic continuation, vanishes in
the strip for | Im s| → +∞ as


ˆΩ(1 − s) = O |Im s| [−][ζ] [0] [�], (31)
�


where ζ 0 - 1/2 + δ 0 . The parameters µ 0 and δ 0 fulfill the constraints µ 0 ∈ (0, min {1, α 0 })
and δ 0 ∈ (α k 1, α k 2 ). The parameter α k 1 is α 0 if α 0 is not an odd natural number, or if α 0 is an
odd number and n 0 does not vanish; otherwise α k 1 coincides with the parameter α k 0 which is
defined in Sec. 4 The index k 2 is the least natural number that is larger than k 1 and such that
α k 2 is not an odd natural number, or it is odd and n k 2 does not vanish. Under the conditions
requested above, the singularity of the function K [ˆ] (s) in s = α k 1 provides Eqs. (20)-(23). The
asymptotic forms (24)-(28) are obtained for the second class of SDs via the analysis performed
in Refs. [34, 33]. The increasing (decreasing) behavior of the bath energy over long times is
obtained for negative (positive) values of the parameters u 0, u [′] 0 [,][ u] [1] [,][ u] 1 [′] [,][ u] [2] [,][ u] [′] 2 [. In this way,]
the conditions on the ohmicity parameter that provide increasing or decreasing behavior of
the bath energy for t ≫ 1/ω s are obtained. This concludes the demonstration of the present
results.


10


### **References**


[1] G. Guarnieri, C. Uchiyama and B. Vacchini, Phys. Rev. A **93**, 012118 (2016).


[2] G. Guarnieri, J. Nokkala, R. Shmidt, S. Maniscalco and B. Vacchini, Phys. Rev. A **94**,
062101 (2016).


[3] J. Jing, D. Segal, B. Li and L.-A. Wu, Sci. Rep. **5**, 15332 (2015).


[4] H.-P. Breuer, E.-M. Laine and J. Piilo, Phys. Rev. Lett. **103**, 210401 (2009); E.-M. Laine,
J. Piilo and H.-P. Breuer, Phys. Rev. A. **81**, 062115 (2010).


[5] Z. He, J. Zou, L. Li and B. Shao, Phys. Rev. A **83**, 012108 (2011).


[6] F.F. Fanchini, G. Karpat, L.K. Castelano and D.Z. Rossatto, Phys. Rev. A **88**, 012105
(2013).


[7] P. Haikka, T.H. Johnson and S. Maniscalco, Phys. Rev. A **87**, 010103(R) (2013).


[8] C. Addis, F. Ciccarello, M. Cascio, G.M. Palma and S. Maniscalco, New J. Phys. **17**,
123004 (2015).


[9] F. Giraldi, Phys. Rev. A **95**, 022109 (2017).


[10] V.G. Morozov, S. Mathey and G. Röpke, Phys. Rev A **85**, 022101 (2012).


[11] V.V. Ignatyuk and V.G. Morozov, Condens. Matter Phys. **16**, 34001 (2013).


[12] V.V. Ignatyuk and V.G. Morozov, Phys. Rev. A **91**, 052102 (2015).


[13] F. Giraldi, arXiv: 1612.03690v1.


[14] C. Addis, G. Brebner, P. Haikka and S. Maniscalco, Phys. Rev. A **89** 024101 (2014).


[15] C.Addis, B. Bylicka, D. Chruscinski and S. Maniscalco, Phys. Rev. A **90** 052103 (2014).


[16] H.-P. Breuer and F. Petruccione, The Theory of Open Quantum Systems, Oxford University Press, Oxford (2002).


[17] U. Weiss, Quantum Dissipative systems, 3rd ed. World Scientific, Singapore (2008).


[18] J. Luczka, Physica A **187** 919 (1990).


[19] G.M. Palma, K.-A. Suominen and A.K. Ekert, Proc. R. Soc. London, Ser. A **452** (1996)
567.


[20] J.H. Reina, L. Quiroga and N.F. Johnson, Phys. Rev. A **65** 032306 (2002).


[21] K. Kraus, States, Effects, and Operations, Lecture Notes in Physics, Vol. **190**, Springer,
Berlin (1983).


[22] V.B. Braginsky and F. Ya Khalili, Quantum Measurements, Cambridge University Press,
Cambridge (1992).


[23] A.J. Leggett, S. Chakravarty, A.T. Dorsey, M.P.A. Fisher, A. Garg and W. Zwerger,
Rev. Mod. Phys. **59**, 1 (1987).


[24] P. Pechukas, Phys. Rev. Lett. **73**, 1060 (1994).


[25] P. Stelmachovic and V. Buzek, Phys. Rev. A 64, 062106 (2001).


11


[26] H. Grabert, P. Schramm and G.L. Ingold, Phys. Rep. **168**, 115 (1988).


[27] L.D. Romero and J.P. Paz, Phys. Rev. A **55**, 4070 (1997).


[28] M.A. Cirone, G. De Chiara, G. M. Palma, P. Haikka, S. McEndoo and S. Maniscalco,
Phys. Rev. A **84**, 031602 (2011).


[29] P. Haikka, S. McEndoo, G. De Chiara, G. M. Palma, and S. Maniscalco, Phys. Rev. A
**84**, 031602 (2011).


[30] M.P. Woods and M.B. Plenio, J. Math. Phys. **57**, 022105 (2016).


[31] M. Reed and B. Simon, Methods of Modern Mathematical Physics Vol. 2, Academics
Press, Inc. (1975).


[32] N. Bleistein and R.A. Handelsman, Asymptotic expansion of integrals, Dover Publications, Inc. New York (1975).


[33] R. Wong, Asymptotic approximations of integrals, Academic Press, Boston (1989).


[34] R. Wong and J.F. Lin, J. Math. Anal. Appl. **64**, 173 (1978).


[35] I.S. Gradshteyn and I.M. Ryzhik, Table of Integrals, Series and Products edited by A.
Jeffrey (Fifth Edition), Academic Press, New York (2000).


12


