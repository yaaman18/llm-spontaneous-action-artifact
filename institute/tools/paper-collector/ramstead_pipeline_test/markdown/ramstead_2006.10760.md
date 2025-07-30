---
title: 
author: 
pages: 27
conversion_method: pymupdf4llm
converted_at: 2025-07-30T20:08:20.130034
---

# 

Draft version July 20, 2020
Typeset using L [A] TEX **twocolumn** style in AASTeX63


**PREDICTIONS OF THE** _**NANCY GRACE ROMAN SPACE TELESCOPE**_

**GALACTIC EXOPLANET SURVEY II:**

**FREE-FLOATING PLANET DETECTION RATES** _[∗]_


[Samson A. Johnson,](http://orcid.org/0000-0001-9397-4768) [1] [Matthew Penny,](http://orcid.org/0000-0001-7506-5640) [2] [B. Scott Gaudi,](http://orcid.org/0000-0003-0395-9869) [1] Eamonn Kerins, [3] [Nicholas J. Rattenbury,](http://orcid.org/0000-0001-5069-319X) [4]

[Annie C. Robin,](http://orcid.org/0000-0001-8654-9499) [5] [Sebastiano Calchi Novati,](http://orcid.org/0000-0002-7669-1069) [6] [and Calen B. Henderson](http://orcid.org/0000-0002-7669-1069) [6]


1 _Department of Astronomy, The Ohio State University, 140 West 18th Avenue, Columbus OH 43210, USA_
2 _Department of Physics and Astronomy, Louisiana State University, Baton Rouge, LA 70803, USA_
3 _Jodrell Bank Centre for Astrophysics, Alan Turing Building, University of Manchester, Manchester M13 9PL, UK_
4 _Department of Physics, University of Auckland, Private Bag 92019, Auckland, New Zealand_
5 _Institut Utinam, CNRS UMR 6213, OSU THETA, Universite Bourgogne-Franche-Com´te, 41bis avenue de lObservatoire, F-25000_
_Besan¸con, France_
6 _IPAC, Mail Code 100-22, Caltech, 1200 East California Boulevard, Pasadena, CA 91125, USA_


(Received July 20, 2020; Revised; Accepted)


Submitted to ApJ


ABSTRACT


The _Nancy Grace Roman Space Telescope_ ( _Roman_ ) will perform a Galactic Exoplanet Survey
(RGES) to discover bound exoplanets with semi-major axes greater than 1 au using gravitational
microlensing. _Roman_ will even be sensitive to planetary mass objects that are not gravitationally
bound to any host star. Such free-floating planetary mass objects (FFPs) will be detected as isolated
microlensing events with timescales shorter than a few days. A measurement of the abundance and
mass function of FFPs is a powerful diagnostic of the formation and evolution of planetary systems, as
well as the physics of the formation of isolated objects via direct collapse. We show that _Roman_ will
be sensitive to FFP lenses that have masses from that of Mars (0 _._ 1 _M_ _⊕_ ) to gas giants ( _M_ ≳ 100 _M_ _⊕_ )
as isolated lensing events with timescales from a few hours to several tens of days, respectively. We
investigate the impact of the detection criteria on the survey, especially in the presence of finite-source
effects for low-mass lenses. The number of detections will depend on the abundance of such FFPs as
a function of mass, which is at present poorly constrained. Assuming that FFPs follow the fiducial
mass function of cold, bound planets adapted from Cassan et al. (2012), we estimate that _Roman_
will detect _∼_ 250 FFPs with masses down to that of Mars (including _∼_ 60 with masses _≤_ _M_ _⊕_ ). We
also predict that _Roman_ will improve the upper limits on FFP populations by at least an order of
magnitude compared to currently-existing constraints.


_Keywords:_ gravitational lensing: micro - planets and satellites: detection - space vehicles: instruments



1. INTRODUCTION


Time and again, surprising results have arisen from
searches for planets beyond our Solar System. Indeed,


Corresponding author: Samson A. Johnson

[johnson.7080@osu.edu](mailto: johnson.7080@osu.edu)


_∗_ During the preparation of this manuscript the name of _The Wide_
_Field Infrared Survey Telescope_ was changed to the _Nancy Grace_
_Roman Space Telescope_ .



one of the first planets discovered defined a population
of “hot Jupiters” (e.g., Mayor & Queloz 1995). These
gas giant planets have orbital periods on the order of
days and can have equilibrium temperatures hotter than
many stars (Collier Cameron et al. 2010; Gaudi et al.
2017). The _Kepler_ mission revealed a substantial population of “super-Earths” (L´eger et al. 2009), planets with
radii between that of Earth and Neptune; planets which
have no analog in our Solar System. Strange system
architectures and planet hosts add even more variety,


2 Johnson et al.



including planets in tightly packed systems (Gillon et
al. 2017), planets orbiting both stars of a binary system
(Doyle et al. 2011), planetary systems orbiting pulsars
(Wolszczan & Frail 1992), and planetary systems orbiting stars at the very bottom of the main sequence
(Gillon et al. 2017). There appears to be almost no
physical constraints on where exoplanets may reside.
Despite this diversity, our statistical census of exoplanets remains substantially incomplete. One area of
parameter space that has yet to be fully explored is that
of planetary-mass objects that are unbound from any
host star. A population of free-floating planetary mass
objects (FFPs) in our Galaxy could have two primary
sources. First, such bodies could be formed in relative
isolation. These would essentially be the lowest mass objects assembled through star-formation processes. Second, such objects could form in a protoplanetary disk
initially bound to a host star, and later become liberated from their host. Regardless of their origin, we will
refer to objects with masses comparable to planets that
are not bound to any host as FFPs.
There are several mechanisms that could lead to the
formation of isolated low-mass stellar objects (see Luhman 2012, and references therein). Stellar cores can be
formed at a range of masses through either gravitational
or turbulent compression and fragmentation (Bonnell et
al. 2008). Here, the lowest mass cores would result in
the lowest mass compact objects; this process may extend down to planetary-mass objects. Alternatively, the
accretion of gas onto a protostellar core can be truncated, e.g, by being dynamically ejected from their birth
clouds by other cores, or by radiation from nearby hot
stars that photoevaporate the envelope from around the
forming star (e.g., Bate 2009).
Photometric surveys of star forming regions can constrain populations of such low-mass stellar objects (e.g.,
Gagn´e et al. 2017). These surveys are most sensitive
to young objects that have not had time to radiate
away their thermal energy from formation and thus remain luminous. In the field, the first thirteen classdefining Y dwarfs were discovered by Cushing et al.
(2011) and Kirkpatrick et al. (2012) using the _Wide-field_
_Infrared Survey Explorer_ (Wright et al. 2010). Modelled
masses for these objects are of order tens of Jupiter
masses ( _M_ Jup ). Volume-limited searches for ultra-cool
field dwarfs (e.g., Bardalez Gagliuffi et al. 2019) constrain these populations, but their low luminosities limit
the number of detections and thus the statistical power
of these surveys. Furthermore, these surveys are unlikely to be sensitive to planets with masses substantially
smaller than that of Jupiter, regardless of their ages.



On the other hand, if the dominant reservoir of FFPs
is a population of previously bound planets, there is no
shortage of methods to liberate them from their hosts.
Planets can be ejected from their systems by the chaotic
processes that occur during planet formation (e.g., Rasio & Ford 1996), stripped from their stars by stellar fly-bys (e.g., Malmberg et al. 2011), or become unbound during the post-main sequence evolution of their
hosts (e.g., Adams et al. 2013). Hong et al. (2018) predict that planet-planet dynamical interactions could also
eject lunar-mass satellites of these planets during the encounters. It is important to emphasize that objects in
the very lowest-mass regime ( _<_ 1 _M_ Jup ) are very difficult
to detect by any radiation they emit, even when they
are young (Spiegel & Burrows 2012).
A robust method to detect isolated planetary mass objects is gravitational microlensing (Di Stefano & Scalzo
1999). A microlensing event occurs when a massive
body (the lens) passes in front of a background star (the
source) within roughly one angular Einstein ring radius
_θ_ E of the lens,
_θ_ E = � _κMπ_ rel _._ (1)


Here, _M_ is the mass of the lensing body, the constant
_κ_ = 4 _G_ ( _c_ [2] au) _[−]_ [1] = 8 _._ 14 mas _M_ _⊙_ _[−]_ [1] [, and the lens-source]
relative parallax is _π_ rel = 1au � _D_ L _[−]_ [1] _−_ _D_ S _[−]_ [1] �, where _D_ L
and _D_ S are the distances from the observer to the lens
and source, respectively.
When the angular separation of the lens and source
is comparable to or smaller than _θ_ E, the background
source is significantly magnified. The duration of an
event is characterized by the microlensing time scale
_t_ E = _θ_ E _/µ_ rel . Thus the size of the Einstein ring in combination with the lens-source relative proper motion ( _µ_ rel )
dictates the duration of the event, which can last from
a few hours to a few hundred days, depending on the
values of the above variables. The primary reason why
microlensing is a powerful technique to detect FFPs is
that it does not rely on the detection of any light from
these essentially dark lenses.
While the phenomenology of typical microlensing
events (for which _θ_ E is much greater than the angular source size) is well understood, that of microlensing events due to low-mass objects has not been frequently discussed. We therefore include a short review
of the phenomenology of low-mass microlensing (specifically when the angular source size is larger than _θ_ E ) in
Appendix A.
One of the pioneering uses of the technique was the
search for the then-viable dark matter candidate Mas
sive Compact Halo Objects, or MACHOs. At the time,
the typical mass for these candidates for dark matter
was unknown, resulting in the need to design a survey


_Roman_ FFPs 3



that was sensitive to the full range of timescales mentioned above. The major microlensing collaborations included the Exprience pour la Recherche d’Objets Sombres (EROS; Renault et al. 1997, the MACHO collaboration (Alcock et al. 1997), the Microlensing Observations in Astrophysics Collaboration (MOA-I, Muraki et
al. 1999 and the Optical Gravitation Lens Experiment
(OGLE-I, Udalski et al. 1992). These collaborations set
out to detect these MACHOs by monitoring the Large
Magellanic Cloud, searching for microlensing events in
this high density stellar source environment, with a large
cross-section through the dark matter halo. Particularly
relevant to this discussion, the combined analysis of the
MACHO and EROS surveys demonstrated that ≲ 25%
of the dark halo is made of planetary-mass MACHOs in
the mass range between roughly 0 _._ 3 times the mass of
Mars and the mass of Jupiter, the first such constraints
on the abundance of planetary-mass objects in halo of
our Galaxy (Alcock et al. 1996). See Moniez (2010) for
a comprehensive history of these efforts.
Once MACHOs were largely ruled out as a dark
matter candidate, microlensing surveys began to focus on lines-of-sight toward the Galactic bulge to constrain Galactic structure (Paczy´nski 1991) and search
for bound exoplanets (Mao & Paczynski 1991; Gould
& Loeb 1992). Initially, these surveys lacked the field
of view to both find relatively rare microlensing events
and monitor them with sufficient cadence to detect
the much shorter (and unpredictable) planetary perturbations. Instead, a two-tier system was employed,
wherein the survey teams used relatively low-cadence
observations to alert follow-up observers of ongoing microlensing events. The relatively small numbers of ongoing microlensing events could then be monitored at
much higher cadence by collaborations with access to a
longitudinally-distributed suite of telescopes. See Gaudi
(2012) for a review of the history of microlensing surveys
for exoplanets during this phase of the field.
Eventually, the MOA and OGLE surveys, along with
the (more recently formed) Korea Microlensing Network
(KMTNet, Kim et al. 2016) survey, have developed the
capability to monitor the Galactic bulge with sufficient
cadence to simultaneously detect isolated microlensing
events and search for perturbations due to bound planets. This resulted in the first tentative detection of an
excess of _∼_ 1 day long events, which implied a substantial population of Jupiter-mass FFPs with an inferred abundance of roughly two free-floating Jupitermass planets per star in the Galaxy (Sumi et al. 2011).
This result was later challenged by Mr´oz et al. (2017),
who placed an upper limit of ≲ 0 _._ 25 Jupiter-mass FFPs
per star. Notably though, Mr´oz et al. (2017) did find



tentative evidence of an excess of very short timescale
events ( _t_ E ≲ 0 _._ 5 d), possibly indicating a population
of free-floating or wide-separation Earth-mass planets,
although it is important to note that these events were
generally poorly sampled and thus have large uncertainties in their timescales. They therefore may be spurious.
Regardless, these efforts demonstrate the potential of
Galactic bulge microlensing surveys to find free-floating
or widely-bound planetary-mass objects.
Indeed, quite recently, multiple well-characterized,
extremely-short microlensing events have been discovered. Mr´oz et al. (2018a), Mr´oz et al. (2019a), and
Mr´oz et al. (2020) together report a total of four FFP
candidates, two of which had timescales consistent with
Earth- or Neptune-mass lenses. Han et al. (2020) report the discovery of three events consistent with brown
dwarf mass lenses (masses _∼_ 0 _._ 04 _M_ _⊙_ ), of which two are
isolated and one is in a near equal-mass binary. An important caveat for candidate FFP events is the potential
to exclude of any potential host stars. If the separation
of a planet and its host is sufficiently large (≳ 10 au; Han
et al. 2005) and the geometry is correct, the source can
appear to be magnified by an effectively isolated planet.
Thus, wide-separation planets can masquerade as FFPs
in a subset of microlensing events.
This has been discussed before by several authors
(Di Stefano & Scalzo 1999; Han & Kang 2003; Han et
al. 2005), all of which propose pathways to determine
whether a planetary mass lens is bound or free-floating.
Mr´oz et al. (2018a) and Mr´oz et al. (2019a) place limits
on the presence of a host photometrically, but detailed
modelling of the magnification curve and photometric
follow-up can also be used to determine if the lens is
isolated (Han & Kang 2003; Han et al. 2005; Henderson
& Shvartzvald 2016). As an example, detailed modeling
has been used to determine the true, bound nature of
an FFP candidates by Bennett et al. (2012) and Han et
al. (2020).
Keeping in mind these caveats, it has been demonstrated previously (Bennett & Rhie 2002; Strigari et al.
2012; Penny et al. 2013; Ban et al. 2016; Henderson &
Shvartzvald 2016; Penny et al. 2017) that a space-based
microlensing survey will have unprecedented sensitivity
to short-timescale microlensing events due to FFP lenses
that have masses comparable to our Moon or greater.
We investigate this opportunity more fully here, as applied to the NASA’s next flagship mission, the _Nancy_
_Grace Roman Space Telescope (Roman)_ .


1.1. _The Nancy Grace Roman Space Telescope and its_
_Galactic Exoplanet Survey_


4 Johnson et al.



Initially called the _Wide Field Infrared Survey Tele-_
_scope_ ( _WFIRST_, Spergel et al. 2015), _Roman_ is currently planned to conduct three Core Community Surveys: the High Latitude Survey (Troxel et al. 2019),
the Type Ia Supernovae Survey (photometric (Hounsell
et al. 2018) and spectroscopic), and the Galactic Exoplanet Survey (Penny et al. 2019). These surveys will be
accompanied by a Guest Observer program (including
notionally 25% of observing time) and a demonstration
of numerous new-to-space technologies with the Coronagraph Instrument (CGI, Debes et al. 2016; Bailey et
al. 2019).
The surveys currently have notional designs that will
allow them to make key measurements that will in turn
provide unique constraints on the nature and time evolution of dark matter and dark energy, as well as provide
novel constraints on the demographics of cold exoplanets
(Akeson et al. 2019). The designs of these surveys are
notional in that the final observing program will not be
settled on until much closer to launch, and, importantly,
will incorporate community input.
For the Roman Galactic Exoplanet Survey, _Roman_
will use the microlensing technique to search for bound
planets with mass roughly greater than that of Earth
( _M_ _⊕_ ) with semi-major axes in the range of _∼_ 1 _−_ 10 Astronomical Units (au) [1] . At planet-host star separations
roughly equivalent to the Einstein radius of the lens system (and thus peak sensitivity), _Roman_ will be able
to detect planets with masses as low as roughly twice
the mass of the Moon, roughly the mass of Ganymede
(Penny et al. 2019, hereafter Paper I). Through finding
these planets near and beyond the water snowline of host
stars, _Roman_ will complement the parameter space surveyed by _Kepler_ (Borucki et al. 2010). When combined,
these broad, monolithic surveys promise to provide the
most comprehensive view of exoplanet demographics to
date, and thus provide the fundamental empirical data
set by which predictions of planet formation theories can
be tested (Penny et al. 2019).
The current version of the _Roman_ microlensing survey area covers approximately 2 deg [2] near the Galactic bulge, comprised of 7 fields covered by the 0.282
deg [2] field-of-view of the Wide Field Instrument (WFI,
Spergel et al. 2015). Throughout the survey, it will
observe some _∼_ 50,000 microlensing events of which
roughly 1400 are predicted to show planetary perturbations (Paper I). The current notional survey design
includes six 72-day seasons, clustered near the begin

1 _Roman_ will also discover ≳100,000 planets with periods ≲64 days
using the transit technique (Montet et al. 2017).



ning and end of the 5 yr primary lifetime of the mission.
Each season will be centered on either the vernal or au
tumnal equinoxes, when the Galactic bulge is visible by

_Roman_ .

During a season, _Roman_ will perform continual observations using its wide 1–2 _µ_ m _W146_ filter at 15 min
cadence. Each visit will have a 46.8 sec _W146_ exposure of the WFI that will reach a precision of 0.01 mag
at _W146_ _≈_ 21. These observations will be supplemented
with at least one and likely two narrower filters (yet to
be decided), which will sample the fields at much lower
cadence. Paper I assumed observations with only one
additional ( _Z087_ ) filter with a 12 hr cadence, but this
observing sequence has not yet been finalized. When a
microlensing source star is sufficiently magnified and observations are taken in more than one filter, _Roman_ will
be able to measure the color of the microlensed source

star. Measurement of the source color and magnitude
can be used to constrain the angular radius of the source
star _θ_ _∗_, which can be be used to measure _θ_ E if the event
exhibits finite-source effects (Yoo et al. 2004). For more
details on the currently planned _Roman_ hardware, the
microlensing survey design, and the bound planet yield,
the reader is encouraged to read Paper I.


1.2. _Constraining the Abundance of Free-Floating_
_Planets with Roman_


The properties of _Roman_ and the Galactic Exoplanet
Survey design that make it superb at detecting and characterizing bound planets are the same properties that
allow it to detect and characterize FFPs. FFPs can

produce events lasting from _∼_ hr to _∼_ day. Many of the
same observables for bound planet microlensing events
are also desirable for FFPs, such as the source color and
brightness, which can constrain the angular source size,
and the mass of the lensing body. Measuring the mass
of an isolated lens requires additional measurements of
event parameters (Gould & Welch 1996), and would
require supplementary and simultaneous ground-based
or space-based observations (e.g., by _EUCLID_, Zhu &
Gould 2016; Bachelet & Penny 2019; Ban 2020). We do
not address parameter recovery through modelling, nor
mass estimation of detected lenses, both of which are
beyond the scope of this work.
The goal of this work is to predict _Roman_ ’s ability
to measure the distribution of short-timescale events attributed to free-floating planets. To do so, we will briefly
revisit the microlensing survey simulations presented in
Paper I and detail the changes we made to them in Section 2. We then examine light curves _Roman_ will detect
in Section 3. Section 4 will contain a discussion of the

yield and limits _Roman_ will place on FFPs in the Milky


_Roman_ FFPs 5



**Table 1.** _Roman_ Galactic Exoplanet Survey Parameters


Area 1.97 deg [2]


Baseline 4.5 years

Seasons 6 _×_ 72 days

Fields 7

Avg. Slew and Settle 83.1 s
Primary ( _W146_ ) filter 0.93-2.00 _µ_ m

exposure time 46.8 s

cadence 15 minutes

total exposures _∼_ 41 _,_ 000 per field
Secondary ( _Z087_ ) filter 0.76-0.98 _µ_ m

exposure time 286 s

cadence ≲ 12 hours
total exposures _∼_ 860 per field

Phot. Precision 0.01 mag @ _W146_ _∼_ 21 _._ 15
**Notes** : A summary of the Cycle 7 design is fully detailed in
Paper I. This is the current design, and is subject to change
prior to the mission. For example, the exposure time and
cadence of observations in the _Z087_ and other filters has
not been set; we have assumed a 12 hour cadence here, but
observations in the other filters are likely to be more frequent.


Way. Finally, we will discuss our findings and conclude
in Sections 5 and 6. We include two appendices, one
that provides a primer on the phenomenology of microlensing events in the regime where the angular size
of the source is much greater than the angular Einstein
ring radius (Appendix A) and a second exploring the
sensitivity of _Roman_ ’s yield to the detection criteria we
impose (Appendix B).


2. SIMULATIONS


To simulate the _Roman_ microlensing survey we use the
free-floating planet module of the GULLS microlensing
simulator (Penny et al. 2013, 2019). Here we only briefly
discuss how FFP simulations differ from the bound
planet simulations of Paper I. We use the mission and
survey parameters for the Cycle 7 design as fully detailed
in Paper I and summarized in Table 1.
GULLS simulates individual microlensing events by
combining pairs of source and lens stars drawn from a
population synthesis Galactic model (GM). We use the
same GM as Penny et al. (2013) and Paper I, version
1106 of the Besan¸con model, for consistency between our
results. Version 1106 is intermediate between the two
publicly available Besan¸con model (Robin et al. 2003,
2012), and is described fully in Penny et al. (2013) and
Paper I. The usefulness of population synthesis GMs
for microlensing was first demonstrated by Kerins et al.
(2009). An updated model by Specht et al. (2020) has
recently been shown to provide a high level of agreement



with the 8,000-event OGLE-IV event sample of Mr´oz et
al. (2019b).
GULLS simulates _Roman_ ’s photometric measurements by injecting GM stars, including the source, into
a synthetic postage stamp image. From this image the
photometric precision as a function of magnification is
computed assuming a 3 _×_ 3 pixel square aperture centered on the microlensing event.
The actual _Roman_ photometric pipeline will be much
more sophisticated than this, using both point spread
function (PSF) fitting and difference image analysis to
perform photometry. Aperture photometry is likely
somewhat conservative relative to PSF fitting photometry in terms of photon noise, but this is offset by optimism in not dealing with relative pixel phase offsets with
an undersampled PSF (see Paper I for a full discussion).
The model microlensing light curve is computed from a
finite-source point lens model (Witt & Mao 1994) with
no limb-darkening. The realistic, color-dependent redistribution of surface brightness from limb darkening will
modify the light curve shape of events in which finitesource effects are present (Witt 1995; Heyrovsk´y 2003),
but does not significantly affect detection probability.
We briefly discuss the impact of omitting limb darkening from our simulations in Section 5.5.
Our simulations follow those of Paper I almost exactly, but we replace the stellar lenses drawn from the
catalogs generated from the GM with an isolated planetary mass object and assume zero flux from the injected lens. This results in all simulated events having
planetary-mass point lenses with the velocity and distance distributions of stars in the GM. One might expect small differences in the phase-space distributions
between stars and FFPs, depending on their origin, but
we do not account for this in this study (e.g., van Elteren
et al. 2019, found FFPs are ejected from clusters with
larger velocities than escaping stars, but only by a few
km s _[−]_ [1], which is much less than typical _∼_ 100 km s _[−]_ [1]

relative velocities between lens and source).
The source and lens of each simulated microlensing
event are drawn from GM catalogs that represent a
0 _._ 25 _×_ 0 _._ 25 deg [2] area of sky, which we call a sight line.
Each event _i_ is assigned a weight _w_ _i_ proportional to the
event’s contribution to the total event rate along a sight
line,


_w_ _i_ = 0 _._ 25 [2] deg [2] _f_ 1106, _Roman_ Γ deg 2 _T_ sim _u_ 0 _,_ max _,i_ 2 _µ_ rel _,i_ _θ_ E _,i_ _,_

_W_

(2)
where _T_ sim = 6 _×_ 72 d is the total _Roman_ microlensing
survey duration, _u_ 0 _,_ max,i is the maximum impact parameter for each simulated event, Γ deg 2 is the sight-line’s
microlensing event rate per square degree, _f_ 1106, _Roman_ is


6 Johnson et al.



a correction factor, and _W_ is a normalization factor defined below. The Γ deg 2 event rates were calculated by
Monte Carlo integration using catalogs of source and
lens stars drawn from the GM.

We use the same _f_ 1106, _Roman_ = 2 _._ 81 as in Paper I which
matches the GM’s event rate to the microlensing event
rate measured using red clump source stars by Sumi et
al. (2013) and corrected by Sumi & Penny (2016). Mr´oz
et al. (2019b) measured microlensing event rates with
a larger sample of events from the OGLE-IV survey.
They measured the event rate per star for source stars
brighter than _I <_ 21 (a so-called all-star event rate) that
was consistent with the Sumi & Penny (2016) red-clump
event rate, but which was a factor of 1.4 smaller than
MOA’s all star event rate estimated by Sumi & Penny
(2016) with sources brighter than _I <_ 20. We elect to
maintain the same event rate scaling as Paper I because
the origin of the discrepancy in all-star rates between
Mr´oz et al. (2019b) and Sumi & Penny (2016) is not
clear, and for reasons discussed in Paper I we expect
that the small bar angle in the GM may cause an overcorrection if corrections are tied to all-star event rates.
For each event, _u_ 0 is uniformly drawn from [0, _u_ 0 _,_ max ],
where _u_ 0 _,_ max _,i_ = max(1 _,_ 2 _ρ_ ) and _ρ_ = _θ_ _∗_ _/θ_ E is the angular radius of the source star relative to _θ_ E . We impose the 2 _ρ_ alternative to ensure that all lens transiting
source events are simulated. We also ran supplemental
simulations at higher masses with _u_ 0 _,_ max _,i_ = max(3 _,_ 2 _ρ_ )
and found consistent event rates to those use _u_ 0 _,_ max _,i_ =
max(1 _,_ 2 _ρ_ ). This event weight should be normalized to
the stellar-lens event rate, so we divide _θ_ E _,i_ by the mass
ratio of the injected lens ( _M_ _p,i_ ) and the star that it is replacing ( _M_ _∗,i_ ), _[√]_ ~~_q_~~ _i_ = ~~�~~ _M_ _p,i_ _/M_ _∗,i_, to correct the value.

We note that this methodology is equivalent to the assumption that there is one FFP per star in the Galaxy.
The normalization factor is then defined



to random sampling and accounting for unequal event
rates is less than 0.1%.


2.1. _Detection Criteria_


We use two detection criteria for microlensing events.
The first is the difference in _χ_ [2] of the observed lightcurve
relative to a flat (unvarying) light curve fit


∆ _χ_ [2] = _χ_ [2] Line _[−]_ _[χ]_ [2] FSPL _[,]_ (4)


where _χ_ [2] Line [is the] _[ χ]_ [2] [ value of the simulated light curve]
data for a flat line at the baseline flux and _χ_ [2] FSPL [is the]
same but for the simulated data to the true finite-source
point lens model of the event.
The second criteria is that _n_ 3 _σ_, the number of consecutive data points measured at least 3 _σ_ above the baseline
flux, must be greater than 6, i.e.,


_n_ 3 _σ_ _≥_ 6 _._ (5)


This criteria serves two purposes. First, it mimics the
type of selection cut that previous free-floating planet
searches have used to minimize the number of false
positives caused by multiple consecutive outliers from
long-tailed uncertainty distributions (e.g., Sumi et al.
2011; Mr´oz et al. 2017). Second, it ensures that any
events detected will stand a good chance of being modeled with 4 or 5 free parameters without over fitting. Extremely short events and those with large _ρ_ may suffer
from degeneracies where even six data points may be insufficient to correctly model the event (Johnson et al., in
prep.). Naively, the probability of six consecutive data
points (assuming that they are Gaussian distributed)
randomly passing this criterion is _∼_ (1 _−_ 0 _._ 9987) [6] _≈_
10 _[−]_ [18] . Given the number of data points per light curve
( _∼_ 4 _×_ 10 [5] ) and the number of light curves ( _∼_ 2 _×_ 10 [8] ),
we expect a by-chance run of 6 points more than 3 _σ_ to
occur at a rate of _∼_ 4 _×_ 10 _[−]_ [4] per survey. Thus this may
appear to be an overly conservative detection criterion.
However, it is likely that neighboring data points will not
be strictly uncorrelated, and therefore it is important to
adopt conservative detection criteria. For example, a
false positive event may be caused by hot pixels, which
are obviously correlated with time. We briefly discuss
the possibility of contamination by this and other false
positives in Section 5.2. We further motivate these se
lection criteria in the next section.

Our predictions for the yields of detectable freefloating planets are calculated using the weights defined
in Equation 2 modified by a Heaviside step function for
each detection criteria,


_N_ det = � _w_ _i_ _H_ (∆ _χ_ [2] _−_ 300) _H_ ( _n_ 3 _σ_ _−_ 6) _._ (6)


_i_



_W_ = �


_i_



2 _µ_ rel _,i_ _θ_ E _,i_
_._ (3)
~~_√q_~~ _i_



such that the sum of all simulated event weights would
equal the number of events occurring over the survey
duration had each stellar lens not been replaced by an

FFP.

We run two sets of simulations, both of which have
lens masses drawn from log( _M/M_ _⊕_ ) _∈_ [ _−_ 5 _,_ 5] (i.e., 0.5%
the mass of Pluto to 0.3 _M_ _⊙_ ). In the first set of simulations, we simulate equal numbers of planets with a
range of discrete masses uniformly spaced by 0.25 dex.
In the second set, we draw log-uniform random-mass
lenses from the same range. In both cases we draw
events until the error on the estimated event rate due


_Roman_ FFPs 7



3. LIGHTCURVES OF FREE-FLOATING PLANETS

AS SEEN BY ROMAN


The continual coverage provided by _Roman_ enables
the detection of the microlensing events caused by freefloating planets without the difficulties faced by groundbased microlensing surveys. In this section we explore
the light curves of free-floating planets that _Roman_
might detect, covering a wide range of planet masses.
We begin with large-mass FFPs and brown dwarfs,
which can be challenging to observe from the ground
due to their event timescales being comparable to several days. Figure 1 shows the light curve for a browndwarf-mass lens in the upper panel. This example has a
relatively long timescale compared to what is expected
for typical free-floating planet events, but we include it
as an extreme case to demonstrate the confusion with

stellar lens events. These cases display the density of
_Roman_ photometry, especially in the lower panel, which
has nearly 1000 3 _σ_ -significant _W146_ measurements in
the time span of roughly 6 days. These events will be
extremely well characterized, and are nearly guaranteed
to have color measurements while the source is magnified.
Figure 2 show the light curves of events at the opposite end of the detectable FFP mass spectrum. A
very low-mass lens exhibits modest finite-source effects
in the upper panel. Much stronger finite-source effects
are apparent in the lower panel for a giant source with
_ρ ≈_ 10. In the latter case, the magnification saturates
at the expected value of 1 + 2 _/ρ_ [2] (Equation A2), i.e.,
just 1 _._ 02 in the absence of limb darkening. To demonstrate the impact of limb darkening for this event, we use
the same event parameters to recompute the magnification using the Lee et al. (2009) method as implemented
in `MulensModel` (Poleski & Yee 2018). This is shown
as the gray, long-dashed line underlying the simulated
event. The peak is higher than in the event without
limb darkening, and the “shoulders” of the top hat drop
modestly. Even for such extreme finite-source events the
impact of limb darkening will be modest on the number
of events that pass selection cuts. Both events highlight
the precision of _Roman_ photometry. The light curves in
Figures 1 and 2 are chosen to demonstrate a number of
morphologies, photometric precisions, masses, and detection significances of _Roman_ events.
For a broader, more representative look at the events
_Roman_ will detect, Figure 3 displays an ensemble of light
curves for each of the five discrete mass lenses we consider. In each panel, we randomly select 100 events that
passed our detection criteria in ∆ _χ_ [2] and _n_ 3 _σ_ . We then
normalize the transparency of each curve to the maximum weight of those events included (Equation 2). In



_M_ _p_ =56 _M_ Jup, _ρ_ =0.00066, _t_ E =11.3 days, _f_ _s_ =0.23, _u_ 0 =0.44







_M_ _p_ =1.8 _M_ Jup, _ρ_ =0.02, _t_ E =1.07 days, _f_ _s_ =0.58, _u_ 0 =0.99

Time [Days]





**Figure 1.** Two examples of simulated events as observed
by _Roman_ . Black (red) points are observations in the _W146_
( _Z087_ ) filters, and the overlying orange line is the input lensing model. Above each panel, _M_ _p_ is the mass of the lens in
Jupiter masses ( _M_ Jup ) or Earth masses ( _M_ _⊕_ ), _ρ_ is the angular size of the source normalized to the Einstein ring, _t_ E
is the Einstein timescale of the event, _f_ _s_ is the blending parameter, and _u_ 0 = _θ_ 0 _/θ_ E is the minimum impact parameter.
We also include the values of log ∆ _χ_ [2] and _n_ 3 _σ_ light curve.
Vertical short-dashed gray lines indicate _±t_ E values of the
event, and the long-dashed grey line the peak of the event.
The expected photometric precision and 15 min cadence for
observations in the primary _W146_ band will make detection
of such events trivial. _Upper left:_ An event with a _∼_ 60 _M_ Jup
brown dwarf lens. _Upper right:_ An event with a _∼_ 2 _M_ Jup
mass lens.


this way, darker curves indicate events that contribute
more to the calculated event rate. We place vertical
dashed lines at the positive/negative weighted average
of _t_ E for these subsets, as well as a horizontal row of
gray dashes below the curves representing the _W146_ cadence (15 min) in the three rightmost panels. Note that
the scales of the horizontal axes shrink with decreasing


8 Johnson et al.



_M_ _p_ =0.10 _M_ _⊕_, _ρ_ =1.47, _t_ E =0.068 days, _f_ _s_ =0.17, _u_ 0 =1.08





22 _._ 175


22 _._ 200


22 _._ 225


22 _._ 250


22 _._ 275


22 _._ 300


22 _._ 325





21 _._ 0 21 _._ 2 21 _._ 4 21 _._ 6


_M_ _p_ =0.59 _M_ _⊕_, _ρ_ =10.26, _t_ E =0.033 days, _f_ _s_ =0.99, _u_ 0 =3.27







7 _._ 4 7 _._ 6 7 _._ 8 8 _._ 0 8 _._ 2 8 _._ 4
Time [Days]


**Figure 2.** Same as Figure 1, but for two very low-mass
lenses. We note that, although both events contain one measurement taken in the _Z087_ filter, this is not representative of most low-mass lens events. _Upper:_ Illustrative light
curve due to a roughly Mars-mass FFP, with relatively mild
finite-source effects. _Lower:_ Illustrative light curve due to a
_∼_ 0 _._ 6 _M_ _⊕_ FFP, in this case lensing a giant source, thereby
exhibiting strong finite-source effects. Note that, in this case
the fact that the source is a giant results in nearly no blending and the large value of _ρ_ . In such cases, the magnification
would saturate at 1 + 2 [shown as the orange line in the]
_ρ_ [2]
absence of limb darkening. However, when we include limb
darkening the lightcurve would appear as long-dashed gray
line (for Γ = 0 _._ 4)


mass (as _t_ E _∝_ _M_ [1] _[/]_ [2] ), but we maintain the scale between
the two rightmost panels. At higher masses ( _≥_ 10 [2] _M_ _⊕_ )
the light curves look like one would expect for pointlike sources. As the mass of the lens decreases, a larger
fraction of detected events exhibit finite-source effects
as described in Appendix A.
Figure 4 shows the rightmost panel of 3 with both axes
re-scaled in order to show finer detail for the lowestmass lenses. However, note the magnification axis re


mains logarithmic. Note there are only 5 dashes (5 photometric measurements) during the expected duration
(2 _t_ E ), marked by the vertical gray dashed lines. However, the true duration of these events is often considerably longer. Were there no finite-source effects, the
events of low-mass lenses would often be too short to

accurately model with the 15 min cadence of the _W146_
band, but because the source crossing time for these
sources can be a factor of several times longer than 2 _t_ E,
these events may be well characterized.


3.1. _Detection Thresholds_


Given the potential challenges involved in detecting
short events, we revisit the detection criteria we presented in Section 2 to ensure they fulfill their purpose.
We require that ∆ _χ_ [2] of an event be at least 300 and that
the event has an _n_ 3 _σ_ of at least 6. These thresholds are
similar in nature to the initial cuts placed by Sumi et al.
(2011) and Mr´oz et al. (2017). Both use _n_ 3 _σ_ _≥_ 3 as well
as a statistic _χ_ 3+ = [�] _i_ [(] _[F]_ _[i]_ _[ −]_ _[F]_ [base] [)] _[/σ]_ _[i]_ [ to quantify the]

significance of candidate events, where _F_ _i_ is the _i_ th data
point within an event with uncertainty _σ_ _i_ and _F_ base is
the baseline flux. Sumi et al. (2011) use _χ_ 3+ _≥_ 80 while
Mr´oz et al. (2017) relaxed this to _χ_ 3+ _≥_ 32 due to the
typically higher quality of the OGLE data.
It is not straight forward to compare our cuts to
the _χ_ 3+ criteria of Sumi et al. (2011) and Mr´oz et al.
(2017), however we can consider an extreme case. Imagine an event that barely passes both our criteria with
∆ _χ_ [2] = 300 and _n_ 3 _σ_ = 6, but with a minimal _χ_ 3+ .
This event would have 6 consecutive data points, five
at 3 _σ_ and a single data point at 16 _σ_, making a total
∆ _χ_ [2] = 301. This particular event would then have a
value of _χ_ 3+ = 31, barely failing to pass the Mr´oz et al.
(2017) _χ_ 3+ threshold. More realistic events would have
higher values of _χ_ 3+, therefore our cuts are at least comparable to those used in Mr´oz et al. (2017) but are likely
slightly more stringent. We also expect fewer systematics and less correlated noise in _Roman_ data compared
to that of ground-based surveys.

Sumi et al. (2011) and Mr´oz et al. (2017) follow their
initial cuts with several more to further vet their sam
ples, ensuring each is truly a microlensing event. These
include (among others) the rejection of light curves with
more than brightening event, the rejection of light curves
with poor goodness-of-fit statistics to initial models, and
the rejection of events that did not have the rise or fall of
the event sufficiently sampled. Without a detailed investigation of the uncertainties in the observables, we must
use heuristic cuts to approximate these detailed investigations. We have not implemented these further cuts because our simulations do not contain the false positives


|Roman FFPs|Col2|
|---|---|
|||
|**102****_M⊕_**<br>**10****_M⊕_**<br>**1****_M⊕_**|**0****_._1****_M⊕_**|


_−_ 0 _._ 25 0 _._ 00 0 _._ 25



_−_ 0 _._ 1 0 _._ 0 0 _._ 1



_−_ 0 _._ 1 0 _._ 0 0 _._ 1



10


1

_−_ 2 _._ 5 0 _._ 0 2 _._ 5



_−_ 1 0 1



10


1



_t −_ _t_ 0 [Days]


**Figure 3.** Samples of simulated magnification curves from events detectable by _Roman_ at each mass of 10 [3] _,_ 10 [2] _,_ 10 _,_ 1 _,_ 0 _._ 1 _M_ _⊕_,
from left to right. For each mass, we randomly select 100 events that passed our detection criteria and plot their magnification
curves. The weighted average _±t_ E is indicated by the vertical dashed lines in each panel. Note the horizontal axis scale changes
as mass decreases and the vertical axis uses a logarithmic scale. The black, horizontal tick marks below the curves indicate
the _W146_ cadence; we note that they are only shown for masses of 10 _M_ _⊕_ and below. The transparency of each curve is
proportional to the weight of the event normalized to the maximum weight of events included in the panel. In this way, darker
lines exemplify events that will contribute more to the event rate for that mass bin.





1.5







1.4


1.3


1.2


1.1


1.0


_−_ 4 _−_ 2 0 2 4
_t −_ _t_ 0 [Hours]


**Figure 4.** The rightmost panel of Figure 3, but rescaled
to highlight the finer detail of light curves arising from _∼_
0 _._ 1 _M_ _⊕_ lenses. Note that the magnification remains in logscale, but the horizontal axis has been converted from days
to hours. The vertical dashed lines are the weighted average
_t_ E, which indicate that the Einstein timescales are generally
much shorter than the observed timescales, which are set
by the crossing time of the source when _ρ ≫_ 1. The gray
vertical dashes match the 15 min observing cadence of the
_W146_ band.


they are designed to reject. We do explore the thresholds
we place in Appendix B, and determine scaling relations
to predict how loosening or tightening these thresholds
will impact _Roman_ ’s free-floating planet yield. These relations can also be used to estimate the change in yield
as the microlensing survey design evolves.
We examine how our thresholds of ∆ _χ_ [2] _≥_ 300 and
_n_ 3 _σ_ _≥_ 6 impact the timescale distribution of events in
Figure 5, where we assume delta functions in mass (one
planet per star) for each mass shown. First, we plot the



10 [3]


10 [2]


10





**Figure 5.** The distribution of detected events as a function
of timescale for different lens mass populations. We plot the
distributions as a function of _t_ E as dashed lines. The solid
lines are the distributions as a function of the maximum of _t_ E
or the source half-chord crossing time, 0 _._ 5 _t_ _c_ . These distributions are nearly identical for masses above 10 _M_ _⊕_ because for
these masses typically _t_ E _≫_ 0 _._ 5 _t_ _c_, whereas for lower masses
the timescale is largely set by the source chord crossing time.
For the two lowest masses, we also plot as dotted lines for the
distribution of 2 [1] _[n]_ [3] _[σ]_ _[ ×]_ [ 15min. The vertical dashed line indi-]

cates 3 _×_ the _W146_ band cadence. The cut we impose on _n_ 3 _σ_
(e.g., the dashed vertical line) eliminates events that are formally ‘significant’ according to the ∆ _χ_ [2] criterion, but would
likely be poorly characterized due to the small number of
significant points. Interestingly, as a result of the fact that
the effective event timescale saturates at the source chord
crossing time for low-mass lenses, many events pass our cuts
that would not in the absence of finite-source effects.


10 Johnson et al.



distribution of events as a function of _t_ = max( _t_ E _,_ [1] 2 _[t]_ _[c]_ [)]


as solid lines. Here _t_ _c_ is the source chord crossing time
as defined in Equation A3 [2] .
These distributions are meant to show duration of
events detected. Events that exhibit extreme finitesource effects (and thus have ‘top hat’ light curves)
_t_ E will be less than [1] 2 _[t]_ _[c]_ [ and the event will be longer than]

expected. This will allow for the detection of events that
would not be typically detectable were there no finitesource effects.
Second, we plot the distribution as a function of solely
the _t_ E values of events as dashed lines. There is essentially no difference between these distributions for
lens masses _≥_ 10 _M_ _⊕_, but we see a strong offset between the solid- and dashed-line distributions for the

0 _._ 1 _M_ _⊕_ events. For low-mass lenses, this demonstrates
the previous point that some detected events would have
expected timescales much shorter than would be detectable considering our requirement on _n_ 3 _σ_ .
Finally, for the two lowest masses we show the distribution as a function of _t_ 3 _σ_ = 21 _[n]_ [3] _[σ]_ _[ ×]_ [ 15min, half the]
length of the event while significantly magnified. These
distributions have no events less than 45 min (the vertical, black dashed line), which is indicative of our detection criteria on _n_ 3 _σ_ . For the 0.1 _M_ _⊕_ events, the
event timescale saturates at the source chord-crossing
timescale for many events, pushing the distribution towards longer durations. This is even more enhanced
when considering the distribution while significantly
magnified.
More broadly, we show the detection efficiency as a
function of the microlensing timescale _t_ E in Figure 6.
The black line is the number of detected events relative

to the number of injected events with a given timescale
within a single 72 day season. The typical timescales for
lenses of five different masses are illustrated with vertical
lines. Within a season, _Roman_ will maintain a ≳ 50%
efficiency down to _t_ E _≈_ 1.5 hr. This efficiency would be
proportionately lower if we consider the efficiency over
the entire 5 yr baseline by a factor of (6 _×_ 72 d ) _/_ (5 _×_
365 d ) = 0 _._ 23 if _t_ 0 is uniformly distributed, since the
Galactic bulge will only be observed for a fraction of a

year.


2 Typically the source radius crossing time as defined by _t_ _∗_ = _ρt_ E =
_θ_ _∗_ _/µ_ rel is used as a proxy for the timescale of the event (e.g.,
Skowron et al. 2011), however, we account for non-zero impact
parameter _u_ 0 _,∗_ similar to Mr´oz et al. (2019a). We follow their
definition except we use the variable _t_ _c_ instead for their Equation
(10). See Appendix A.



1.00


0.10


0.01

|Col1|0.1M⊕ 1|M⊕|10M⊕|102M⊕|103M⊕|
|---|---|---|---|---|---|
|||||||



**Figure 6.** _Roman_ ’s detection efficiency as a function of
timescale (the black solid line) computed as the fraction of
the events that pass our detection criteria relative to all
events. _Roman_ will have _>_ 50% detection efficiency down
to events with timescales as short as 1.5 hr. The five vertical lines indicate typical timescales for lenses with the mass
indicated.


Overall, in this section we have demonstrated that
_Roman_ will be able to detect a wide range and variety
of short timescale microlensing events. This will impact
the overall timescale distributions of microlensing events
that _Roman_ will detect, and must need to be accounted
for in determining the detection sensitivity used to infer the true underlying distribution of event timescales,
regardless of the nature of the lenses.
In the next section, we now present our predictions
for the yield of and limits on free-floating planets given
the fiducial Cycle 7 survey design.


4. PREDICTED YIELDS AND LIMITS


In this section, we present our predictions for the number of FFPs _Roman_ will detect, as well as the limits on
the total mass of FFPs that can be set by _Roman_ . Recall that the yields are calculated from summing the
weights of simulated events that pass our detection cuts
using Equation 6. We maintain our detection criteria
of ∆ _χ_ [2] _≥_ 300 and _n_ 3 _σ_ _≥_ 6, but discuss the impact of
changing these in Appendix B.


4.1. _Yield_


We must assume a mass function for FFPs if we are
to estimate the number of FFPs that _Roman_ will find.
We assume two forms of mass function, one log-uniform
in mass,
_dN_
= 1 dex _[−]_ [1] _,_ (7)
_d_ log _M_ _p_

and another inspired by an inferrred mass function of
bound planets detected by microlensing (following Cas

12 Johnson et al.



zoidal rule to integrate the number of detections with
masses from 0 _._ 1 _−_ 1000 _M_ _⊕_ to estimate the total yield
of FFPs. We include rows for FFPs with masses of 0.01
and 10 [4] _M_ _⊕_ for reference. Were the mass function simply log-uniform, nearly 1000 free-floating planets would
be detected. In the case of the fiducial mass function,
we predict that _Roman_ will detect roughly 250 FFPs.
Next, we consider how these populations will manifest in the timescale distribution of microlensing events
measured by _Roman_ . To start, we show the expected
timescale distribution of detected stellar events with the
same detection criteria (∆ _χ_ [2] _≥_ 300 _, n_ 3 _σ_ _≥_ 6) in Figure
7. Note that the minimum mass included in the Galac
tic Model is 0 _._ 08 _M_ _⊙_ _≈_ 80 _M_ Jup in the Galactic disk and
0 _._ 15 _M_ _⊙_ in the Galactic bulge. Then we consider three
cases for populations of FFPs. The blue hatched region has an upper boundary that reflects the limit of
at most 0.25 Jovian planets per star from Mr´oz et al.
(2017). We also include consider the population of 5_M_ _⊕_ free-floating or wide-separation planets that Mr´oz
et al. (2017) cautiously consider as a possible explanation of the excess of very short-timescale events. The
orange shaded region has a lower (upper) bound corresponding to 5 (10) FFPs per star in the MW that are
5- _M_ _⊕_ . Thirdly, we show the expected distribution of
detections using the continuous fiducial mass function
in red. We also draw a realization of this mass function

which is included as the gray histogram with Poisson
error bars.
If our fiducial assumptions are reasonable, _Roman_
will be able to detect the signature of terrestrial mass
to Jovian mass lenses in the event timescale distribu
tion. With the lowest mass planets giving rise to events
with extended timescales due to finite-source effects,
the sensitivity is pushed to events with lens masses as
low as a few times that of Mars. The fiducial mass
function we use produces events detectable by _Roman_
with timescales stretching over three orders of magnitude. These will leak into the timescale distribution

attributable to the stars in the Galaxy, but because the
model truncates at 0.08 _M_ _⊙_ there is no smooth transi
tion.


4.2. _Limits_


If _Roman_ detects no free-floating planets in a given
mass range, it can still place interesting constraints on
the occurrence rate of such planets, which in turn can
be used to constrain planet formation theories. We can
place expected upper limits on populations of FFPs using Poisson statistics, following Griest (1991). If we return to our delta function mass distribution such that

we assume there is one planet of that mass per star,



**Table 2.** Expected Free-Floating Planet Yields


Mass Mass Function


( _M_ _⊕_ ) One-Per-Star Log-Uniform Fiducial


0.01 1 _._ 22 0 _._ 349 0 _._ 698

0.1 17 _._ 9 5 _._ 13 10 _._ 3

1 88 _._ 3 25 _._ 2 50 _._ 5

10 349 83 _._ 0 103 _._

100 1250 298 68 _._ 9

1000 4100 976 42 _._ 0

10000 13300 3170 25 _._ 4

Total 3750 897 249


Note—The ‘Total’ row is an integration using the
trapezoidal rule from 0 _._ 1 _−_ 1000 _M_ _⊕_ . The first and
last rows are included for reference.


we can place a 95% confidence level upper limit for any
mass bin, which corresponds to the situation in which
we would expect fewer than 3 planets per star [3] . Figure
8 plots the 95% confidence level _Roman_ will be able to
place on the _total mass_ of bodies per star in the MW
composed of bodies of mass _M_ if no lenses of that mass
are detected. Note that the vertical axis is equivalent
to _M_ _p_ _dN/d_ log _M_ _p_ in units of _M_ _⊕_ . For comparison, we
plot our fiducial mass function (Equation 8), and the
mass distribution for Solar System bodies [4] . The latter
is to give some intuition as to if there were an equivalent of a Solar System’s mass function worth of unbound
bodies per star in the MW, but we note that such a
mass function is likely to be incomplete at low masses,
and possibly also at higher masses (Trujillo & Sheppard
2014; Batygin & Brown 2016). In other words, for typical planetary formation scenarios, a higher number of
low-mass objects are ejected than remain in our solar
system, and in least a subset of planetary systems, a
higher number of higher-mass objects are ejected than
remain in our solar system.
This origin of the shapee of the total mass limit curve
deserves some discussion. For FFP masses _M_ ≳ 1 _M_ _⊕_
the curve rises as _M_ [1] _[/]_ [2], which is somewhat counter intuitive, though may be recognized by those familiar with
dark matter microlensing surveys. The number of expected microlensing events _Roman_ will detect is set by


3 More specifically, if one expects 3 planets and detects none, according to the Poisson distribution, one could rule out the hypothesis that there are 3 planets at a significance of 1 _−_ exp( _−_ 3) _≃_
95%.
4 ssd.jpl.nasa.gov


_Roman_ FFPs 13



the microlensing event rate Γ, which scales as the square
root of the object mass Γ _∝_ _M_ [1] _[/]_ [2], _if there is a fixed_
_number of objects_ . But the vertical axis of Figure 8 is
the total mass of expected objects of mass _M_ _p_ per star
_M_ tot, not the total number. So for fixed _M_ tot, the number of objects scales as the inverse of the object mass
_M_ _[−]_ [1] and thus the microlensing event rate produced by
a fixed total mass of object scales with the individual
object mass as _M_ _[−]_ [1] _[/]_ [2] . The total number of detections
therefore scales as _N_ det _∝_ _M_ tot _M_ _p_ _[−]_ [1] _[/]_ [2] The survey limit
is a contour of a constant number of expected detections, and thus the total mass of ejected objected scales
as as _M_ tot _∝_ _M_ [1] _[/]_ [2] .
Below _M_ _∼_ 1 _M_ _⊕_, the finite size of a typical _Roman_
source star becomes larger than the typical Einstein ring
radius of the lens, and so the event rate per object becomes independent of object mass. But the event rate
per total object mass scales as _M_ _[−]_ [1], and we would expect the limit curve to become more steeply positive
and scale as _M_ _[−]_ [1] . However, the transition to the finitesource dominated regime begins to reduce the peak magnification of events, even if lengthening them, which
eventually significantly reduces the probability of a microlensing event being detected. Between _∼_ 0 _._ 01 _−_ 1 _M_ _⊕_,
finite-source effects from events with 1 _< u_ 0 _< ρ_ increase the detectable event rate (and reduce the total
mass limit) by up to a factor of two relative to events
with only _u_ 0 _<_ 1. Below _M_ ≲ 0 _._ 01 _M_ _⊕_ finite-source effects decrease the maximum magnification of microlensing events to the point where they start to become undetectable, and the detection efficiency begins to fall far
faster than the event rate increases, and the slope of the
limit curve inverts and becomes sharply negative.
Viewed broadly, the total mass limit curve shows that
_Roman_ will be an extremely sensitive probe of the _total_
_mass budget_ of loosely bound and free-floating masses.
At its most sensitive mass, _M ∼_ 3 _×_ 10 _[−]_ [2] _M_ _⊕_ (near
the mass of Mercury), _Roman_ would be sensitive to total masses of just _∼_ 0 _._ 1 _M_ _⊕_ per star (or roughly three
objects per star). _Roman_ will be sensitive to a total
mass of 1 _M_ _⊕_ or less of objects with masses over a range
of _∼_ 0 _._ 003 _−_ 100 _M_ _⊕_, or more than 5 orders of magnitude in mass. While for the lowest mass objects these
total masses are large compared to the mass budget of
the present Solar System, they are small compared to
the total mass of planetesimal disks that are required
to form solar-system-like planet configurations in simulations. For one example, the Nice model considers
initial planetesimal disk masses between 30 and 50 _M_ _⊕_
beyond Neptune (Tsiganis et al. 2005). For a broader
view of the expected population of loosely bound and
free-floating objects, we can compare the _Roman_ total



We also include three frequencies from other observational efforts:


1. Sumi et al. (2011) reported that there may be two
free floating Jupiter mass planets per star in the
MW. Although inconsistent with the (more recent)
limits set by Mr´oz et al. (2017), we display this
result for context.



mass limit curve to various predictions and constraints
on these populations.
The first set of comparisons we draw is between _Ro-_
_man_ ’s limits and limits set by microlensing searches for
massive halo compact objects (MACHOs). There are
three studies we consider


1. As mentioned in the introduction, Alcock et al.

(1996) presented combined results of the MACHO
and EROS microlensing surveys. These surveys
were searching for MACHOs as candidates for the
dark matter mass components of the MW halo.


2. Griest et al. (2014) found a similar limit on primordial black holes, but used the _Kepler_ transit
survey. _Kepler_ provides relatively high-cadence
observations of a fixed, relatively dense, stellar
field, which is nearly optimal for a survey of microlensing events. The drawbacks were that this
was towards a relatively low stellar density field
compared to the LMC or the Galactic Center and
that potential sources were much closer than those
of typical of microlensing events. The limits placed
here are from the analysis of 2 years of the _Kepler_
mission, looking for short timescale events.


3. Niikura et al. (2019a) used the Hyper SuprimeCam on the Subaru Telescope (Subaru/HSC) to
perform 2 min cadence observations of M31 with
high resolution. This search yielded the best constraint on low mass primordial black holes as a
component of the Milky Way Dark Matter halo.


4. Niikura et al. (2019b) placed limits roughly 50%
lower that than MACHO+EROS result at 50 _M_ _⊕_
using 5 years of OGLE IV data. We do not include
this result in Figure 8 due to space constraints.


These limits are not meant to be a direct comparison, so
we simply scale their limits by assuming a stellar number
density of _n_ _⋆_ = 0 _._ 14 pc _[−]_ [3] and a dark matter halo mass
density of _ρ_ halo = 0 _._ 3 GeV _/_ cm _[−]_ [3] . We determine their
measured halo mass fractions, _f_ HM, from their figures.
Then, the mass of free-floating objects per star is simply



_._ (9)
�



_f_ HM _ρ_ halo

= 10 [3] M _⊕_
_n_ _⋆_



0 _._ 06
� _f_ HM


14 Johnson et al.



















**Figure 8.** The heavy solid black line shows the 95% confidence upper limit on the total mass of objects per star as a function
of the object mass that _Roman_ will be able to place if no objects of a fixed mass are detected. It is orders of magnitudes
lower than past limits and can test predictions on the abundance of FFPs from planetary formation (or free floating compact
objects formed from other mechanisms, such as primordial black holes). The black dashed lines represent similar limits placed
by microlensing searches for massive compact halo objects. The blue dot-dashed line shows our fiducial mass function (Equation
8). For context, the red dashed line shows the case if roughly a Solar System’s worth of objects per star were free floating in
the Galaxy. The observational results of previous microlensing surveys are plotted in black points indicated by ‘Sumi+ 2011’
and ‘M´roz+ 2017’. The black circles are frequencies for widely separated bound planets reported by Nielsen et al. (2019) using
direct imaging. Upper limits from three related studies are plotted in gray (see Section 4.2 for details).
Citations: Alcock et al. (1996); Sumi et al. (2011); Griest et al. (2014); Ma et al. (2016); Barclay et al. (2017); Mr´oz et al.
(2017); Hong et al. (2018); Niikura et al. (2019a); Nielsen et al. (2019).



2. Mr´oz et al. (2017) place an upper limit of fewer
than 0.25 Jupiter mass FFPs per star in the MW.
This is indicated by the black arrow in the upper
right. Mr´oz et al. (2017) find a tentative signal for
five-to-ten 5 _M_ _⊕_ mass FFP per star in the MW.
This is represented by the black vertical bracket.
Note that if no events occur with a Jupiter mass
lens, then _Roman_ will place a limit of fewer than
one Jupiter mass planet per _∼_ 100 stars in the MW.
This will improve the limit placed by the OGLE
survey from 8 years of data by more than an order
of magnitude (Mr´oz et al. 2017).


3. We consider measurements of the frequencies of
_bound_ planets and brown dwarfs found using direct imaging by Nielsen et al. (2019). While these
are bound planet frequencies, they are for companions with semi-major axes from 10-100 au, which



would likely be mistaken for free-floating planets
in microlensing surveys, and thus provide a useful
comparison. Nielsen et al. (2019) found a 3.5%
occurrence rate for 5-13 Jupiter mass planets and
a much lower rate of 0.8% for 13-80 Jupiter mass
brown dwarfs for hosts with mass 0 _._ 2 _> M/M_ _⊙_ _>_
5. We include these two frequencies as black circles in Figure 8 with vertical errors being their reported uncertainties and horizontal the associated
ranges. _Roman_ will be sensitive to these widely
bound companions, so distinguishing these freefloating planet false positives will be important.


We also plot predictions on the total mass of FFPs
per star from a number of theoretical simulations:


1. Pfyffer et al. (2015) present simulations of formation and evolution of planetary systems, in which
only _∼_ 0.04 _M_ Jup of planets are ejected per star in


_Roman_ FFPs 15



the optimistic case of no eccentricity or inclination
damping.


2. Ma et al. (2016) predict the number of planets
ejected per star from dynamical simulations. We
take values from their models of 0 _._ 3 _M_ _⊙_ stars, in
that 12.5% of stars eject 5 _M_ _⊕_ of mass in 0 _._ 3 _M_ _⊕_
bodies.


3. Barclay et al. (2017) predict the number of planetesimals ejected from systems during planet formation. We only compare our limit to their prediction in which giant planets are present in the
system, as gray horizontal bar spanning the width
of the bins used. In the case that no giant planets
are present, fewer objects are ejected.


4. Hong et al. (2018) predict that _O_ (0 _._ 01 _−_ 1) moons
will be ejected from systems following planetplanet dynamical interactions. We assume these
moons have masses from 0.1-1 _M_ _⊕_, and thus a
range of possibilities is included within the gray
shaded region. This is a generous upper mass limit
compared to the moons of our Solar System, but
we note that that little is understood on the for
mation of exomoons. As an example of an unexpected possibility, there is (contested) evidence
of a Neptune-sized exomoon in the Kepler-1625b
system (Teachey & Kipping 2018; Kreidberg et al.
2019; Teachey et al. 2020).


Thus, we conclude that _Roman_ will not only improve
the constraints on the abundance of objects with masses
from that of less than the moon to the mass of Jupiter
by an order of magnitude or more, but it will also allow for a test of model predictions for the total mass of
ejected planets in several different planet formation and
evolution theories.


5. DISCUSSION


5.1. _Event Detection_


The _Roman_ microlensing survey will record nearly
40 _,_ 000 photometric data points for _∼_ 10 [8] stars over
its 5 yr duration. While we have perfect knowledge
within these simulations, practically finding events due
to very low-mass lenses will likely require more sophisticated search algorithms. Microlensing surveys have used
clear and specific cuts in identifying events. For example, Mr´oz et al. (2017) made a series of detection cuts
based on the temporal distribution of data points during a candidate event, e.g, the number of observations
obtained while the flux is rising and falling. These additional cuts were made in order to avoid false positives
like flares or cataclysmic variables.



Machine learning classifiers are also starting to be applied to microlensing survey data as well. Wyrzykowski
et al. (2015) searched through OGLE-III data using a
random forest classifier. Godines et al. (2019) present a
classifier for finding events for low-cadence wide field
surveys. Khakpash et al. (2019) developed a fast,
approximate algorithm for characterizing binary lens
events. Bryden et al. (in prep.) are developing a machine learning classifier for the microlensing survey being
performed with the United Kingdom Infrared Telescope
(UKIRT, Shvartzvald et al. 2017). This survey is designed to be a pathfinder for the _Roman_ survey, and
is mapping the near-infrared microlensing event rate in
candidate _Roman_ fields.
Still, most of these efforts have focused on the familiar regime of small finite source sizes _ρ ∼_ 10 _[−]_ [2] _−_ 10 _[−]_ [3]

regimes that are most familiar in microlensing surveys.
It will need to be carefully examined how effective these
search techniques are in detecting the extremely shorttimescale events we are considering, particularly those
with qualitatively different morphologies from the more
familiar _ρ ≪_ 1 single lens microlensing events. Here
we use only the ∆ _χ_ [2] and _n_ 3 _σ_ metrics to determine if
an event is detected in these simulations (but see Appendix B). However, events may be detectable by _Ro-_
_man_ over a wider region of parameter space using different event selection filters, including the low-amplitude
top hat events caused by low mass lenses.


5.2. _False Positives_


The full sample of microlensing events detected by _Ro-_
_man_ will need to be vetted for false positives. Detector
artifacts such as hot pixels or other defects may introduce systematics that could mimic a short-timescale microlensing event. However, among the Core Community
Surveys, the Galactic Exoplanet Survey provides one of
the best opportunities to characterize the Wide Field
Instrument H4RG detectors, facilitating the exclusion
of these artifacts in light curves (Gaudi et al. 2019).
Astrophysical sources could also be mistaken for microlensing events. Similar to those false positives of
ground based microlensing surveys, these will include at
least asteroids, cataclysmic variables, and flaring stars.
While a more rigorous event detection algorithm will
allow _Roman_ to mitigate these in the full microlensing
sample, even simple cuts will sometimes suffice. For example, M dwarf flares rarely last longer than 90 minutes
(Hawley et al. 2014) making their exclusion almost guaranteed by a simple cut on _n_ 3 _σ_ . Longer lasting events
such as novae or cataclysmic variables will have many
photometric data points during their eruption to model
and reject them. Spatio-temporal clustering of candi

16 Johnson et al.



date events and astrometric centroid analysis can be
used to recognize asteroids that have not already been
identified prior to a pipeline run.


5.3. _Degeneracies_


As identified in Mr´oz et al. (2017), one event (MOAip-01) in the sample of short timescale events presented
by Sumi et al. (2011) has a degenerate solution. The
event was reported with _t_ E = 0 _._ 73 d, but an alternate
solution with much longer _t_ E = 8 _._ 2 d is favored. In this
case, a larger blending parameter ( _f_ _s_ ) and smaller impact parameter ( _u_ 0 ) resulted in the alternate solution.
The major difference between these models is in the appearance of the wings of the magnification event. This
degeneracy is well described in Wo´zniak & Paczy´nski
(1997). _Roman_ should be able to distinguish between
these approximately degenerate events via its high precision and cadence.

Another relevant degeneracy occurs in lensing events
with large relative angular source size _ρ_ . In this regime,
the magnification over the duration of the event is
roughly constant (in the absence of limb darkening) and
set by Equation A2. As a result, _ρ_ becomes nearly degenerate with the blending parameter _f_ S = _F_ S _/_ ( _F_ S + _F_ B ),
which is the fraction the source flux _F_ S contributes to the
observed baseline flux _F_ S + _F_ B, where _F_ B is the blended
flux. Essentially, the flux from the source alone cannot be confidently measured without precise and dense
photometry, which can be used to distinguish the subtle differences in these broadly degenerate light curves.
Thus, in the presence of blended light, one may underestimate the true peak magnification (Mr´oz et al. 2020,
Johnson et al., in prep.) making it difficult to constrain
_ρ_ precisely. This is important because _ρ_ depends on
the angular source size which will be poorly constrained
when no color measurement is made while the source is
magnified. These measurements will likely not occur for
short timescale events and those that exhibit extreme
finite-source effects. We note that this degeneracy persists even in the presence of limb darkening (Johnson et
al., in prep.)
Since, for fixed _θ_ _∗_, _ρ_ increases as the planet mass (and
thus the _θ_ E ) decreases. Thus as the lens mass decrease,
more and more events enter into the _ρ ≫_ 1 regime,
thereby increasing the likelihood that they will suffer
from this degeneracy. We note that this continuous degeneracy is different than the discrete degeneracy exhibited in the event reported by Chung et al. (2017).
In order to estimate the fraction of events for which
finite source effects should be detectable, in Figure 9
we show the cumulative fraction of detected events as a
function of _ρ/u_ 0 . Events that have _ρ/u_ 0 = _θ_ _∗_ _/θ_ 0 ≳ 0 _._ 5



should exhibit finite-source effects (Gould & Gaucherel
1997). Events that satisfy this criterion and have _ρ ≫_ 1
will be more susceptible to the above _ρ −_ _f_ S degeneracy.
Fortunately, most of our low-mass lenses that exhibit
finite-source effects will be detected in events where the
source star dominates the baseline flux (or have large
values of _f_ S ). This is shown in Figure 10, where we plot
the cumulative fraction of detected events as a function
of _f_ S in _W146_ (upper panel) and _Z087_ (lower panel).
Vertical dashes mark the median values of _f_ S of these distributions, and note that the markers for 10 [2] and 10 [3] _M_ _⊕_
lie on top of each other. We also include the source magnitude distributions for detected events in Figure 11 for
_W146_ (upper panel) and _Z087_ (lower panel). Brighter
sources will contribute most to the low-mass lens event

rates, as one would expect. However, because the fraction of blended flux is not known _a priori_, this argument
can only be used in statistical sense.
Events of all masses will have little blending in _Z087_,
but small mass lenses (that last ≲ 6 hr) will be unlikely have a _Z087_ measurement taken while magnified.
Measurements from multiple filters may allow an estimate of the source color. Table 3 shows the fraction of

events that will have a color measurement taken while
the source is magnified, resulting the breaking of the degeneracy. Only 11% of 0 _._ 1 _M_ _⊕_ lenses will have a color
measurement if the _Z087_ measurements (or other alternative band) have a cadence of 12 hr, but this fraction
can be more than tripled if the cadence increases to 6
hr. At 1.0 _M_ _⊕_, a larger total number of detected events
results in the fraction increase less with an increase in

cadence. We also include fractions of detected events

with color measurements if our threshold _n_ 3 _σ_ _≥_ 6 were
to be relaxed to only 3, making the percentage even
lower for low-mass lenses. Note the modest decrease in

percentages arises from the fact that many more lowmass lens events are detected when the _n_ 3 _σ_ threshold is
relaxed (see Appendix B, especially Figure 14).
Alternatively, this degeneracy may be broken through
the 5 year baseline of the microlensing survey. Potentially blended sources may become apparent as blended
stars (either unrelated stars, the host star if the planet
is actually bound but widely-separated [see Section 5.4],
or a companion to the host star) move away from the
line of sight to the source. This fact will also be used in
constraining the presence of potential host stars to FFP
candidates.
Johnson et al. (in prep.) demonstrate that there is a
second degeneracy in events with _ρ ≫_ 1. This is a multiparameter degeneracy between the effective timescale of
the event, which is well approximated by the time to
cross the chord of the source _t_ _c_, the impact parame

_Roman_ FFPs 17


_f_ s
0 _._ 02 0 _._ 03 0 _._ 06 0 _._ 10 0 _._ 18 0 _._ 32 0 _._ 56



















**Figure 9.** The cumulative fraction of detected events as
a function of _ρ/u_ 0 which is equal to _θ_ _∗_ _/θ_ 0, or the angular source radius relative to the angular impact parameter.
Almost all low-mass lens events detected will exhibit finitesource effects (with _ρ/u_ 0 ≳ 0 _._ 5, Gould & Gaucherel 1997).


ter of the lens with respect to the center of the source
_u_ 0 _,∗_, and the time to cross the angular source radius
_t_ _∗_ = _θ_ _∗_ _/µ_ rel (see Section A for definitions of these quantities). This is easiest to understand in the absence of
limb darkening. A larger impact parameter _u_ 0 _,∗_ results
in a shorter event, but with the same peak magnification
(due to the ‘top hat’ nature of events with _ρ ≫_ 1). But
this shorter duration can be accommodated by scaling
_t_ _∗_ . Since neither _θ_ _∗_ nor _µ_ rel are known _a priori_, it is
impossible to measure _µ_ rel in the regime where these assumptions hold. Johnson et al. (in prep.) demonstrate
that this degeneracy holds for limb darkened sources as
well.

We are investigating the severity and impact of these
degeneracies on the ability to recover event parameters
in events with extreme finite-source effects (Johnson et
al., in prep.).


5.4. _Wide-bound Confusion_


While _Roman_ will have sensitivity to the short time
scale events of FFPs, true FFPs can be confused with
widely separated but bound planets. If a bound planet
has a large enough projected separation, the source may
only be magnified by the planet and not the host star
(Di Stefano & Scalzo 1999; Han & Kang 2003; Han et al.
2005). This confusion requires proper accounting if accurate occurrence rates for both FFPs and wide-bound
planets are to be reached. To this end, Han et al. (2005)
summarize three methods for distinguishing the presence of a host star.



**Figure 10.** Most source stars will contribute the majority
of baseline flux in low-mass lens events. Normalized cumulative distributions of the blending parameter for detected
events among the five mass bins. For each lens mass, a vertical tick on the distribution marks the value of _f_ _s_ at which
half of events have a greater _f_ _s_ . For higher mass lenses, this
value is _f_ _s_ _≈_ 0 _._ 20 and this value only increases as lens mass
decreases. For the 0.1 _M_ _⊕_ lenses, most detected events have
_f_ _s_ _>_ 0 _._ 5 and thus the source makes up the majority of the
baseline flux. _Upper:_ Blending in _W146_ . _Lower:_ Blending
in _Z087_ .



16 18 20 22 24 26

|1.0<br>Magnitude)<br>0.1M⊕<br>W146<br>1M⊕ 103M⊕<br>10M⊕ 102M⊕<br>0.5<br>dN/d(Source<br>1.0<br>0.1M⊕ 103M⊕<br>Z087<br>1M⊕<br>Normalized<br>102M⊕<br>0.5<br>10M⊕|0.1M⊕<br>W146<br>1M⊕ 103M⊕<br>10M⊕ 102M⊕|Col3|Col4|
|---|---|---|---|
|0.5<br>1.0<br>103_M⊕_<br>102_M⊕_<br>10_M⊕_<br>1_M⊕_<br>0_._1_M⊕_<br>_W146_<br><br><br><br><br><br><br>0.5<br>1.0<br>103_M⊕_<br>102_M⊕_<br>10_M⊕_<br>1_M⊕_<br>0_._1_M⊕_<br>_Z087_<br>Normalized_ dN/d_(Source Magnitude)|103_M⊕_<br>102_M⊕_<br>10_M⊕_<br>1_M⊕_<br>0_._1_M⊕_<br>_Z087_|103_M⊕_<br>102_M⊕_<br>10_M⊕_<br>1_M⊕_<br>0_._1_M⊕_<br>_Z087_|103_M⊕_<br>102_M⊕_<br>10_M⊕_<br>1_M⊕_<br>0_._1_M⊕_<br>_Z087_|
|0.5<br>1.0<br>103_M⊕_<br>102_M⊕_<br>10_M⊕_<br>1_M⊕_<br>0_._1_M⊕_<br>_W146_<br><br><br><br><br><br><br>0.5<br>1.0<br>103_M⊕_<br>102_M⊕_<br>10_M⊕_<br>1_M⊕_<br>0_._1_M⊕_<br>_Z087_<br>Normalized_ dN/d_(Source Magnitude)|103_M⊕_<br>102_M⊕_<br>10_M⊕_<br>1_M⊕_<br>0_._1_M⊕_<br>_Z087_|||

Source Magnitude


**Figure 11.** Low-mass lens will have much brighter source
stars. Here we show the source magnitude distributions normalized to their peaks. The higher mass lenses (≳ 10 _M_ _⊕_ )
will have nearly identical distributions, but lower masses
than that will strongly deviate. _Upper:_ Source magnitude
distribution in _W146_ . _Lower:_ Source magnitude distribution in _Z087_ .


The first method was originally described by Han &
Kang (2003), in which the magnification by a bound
planetary mass object will deviate from that of an isolated lens. In this scenario, rather than the effective
point caustic of an isolated lens, the source is magnified










18 Johnson et al.



**Table 3.** Fraction of detected events with color
measurements while source is magnified


Mass _n_ 3 _σ_ _≥_ 6 _n_ 3 _σ_ _≥_ 3


[ _M_ _⊕_ ] 12 hr 6 hr 3 hr 12 hr 6 hr 3 hr


0.1 11% 35% 56% 8% 25% 42%

1.0 12% 23% 37% 11% 20% 34%

10 32% 53% 75% 32% 53% 75%

100 74% 92% 96% 75% 93% 96%

1000 98% 99% 99% 98% 99% 99%


by a planetary caustic which changes the morphology
of the light curve. Han & Kang (2003) assume some
fiducial detection thresholds, and find this method can
distinguish ≳ 80% of events for projected separations
≲ 10au and mass ratios down to _q ≈_ 10 _[−]_ [4] . This deviation was first observed by (Bennett et al. 2012) and
more recently observed in the short (4 day) event reported by Han et al. (2020), where the presence of a
host was determined through 0.03 magnitude residuals
near the peak magnification of a single lens model.
Another pathway to determine if an FFP lens is truly
isolated would be to rule out any magnification from
a photometrically undetected host (Han et al. 2005).
This signal would appear as a long-term, low-amplitude
bump in the light curve. Han et al. (2005) show that
nearly all planets with projected separations of less than
about 13 au will have the presence of their host stars
inferred this way. Assuming the semi-major axis distribution of bound planets is log-uniform, _∼_ 30% of those
with _a ∈_ [10 _[−]_ [1],10 [2] ] lie outside 13 au.
The third method is to directly measure blended light
from a candidate host. This can be performed in earlier or later seasons with _Roman_ by searching for PSF
elongation, color-dependent centroid shifts, or event resolution of the lens host star and the source star. Henderson & Shvartzvald (2016) find _Roman_ can exclude hosts
down to ≳ 0 _._ 1 _M_ _⊙_ depending on the lens distance and
the nature of the source star. Paper I finds that the majority of hosts to bound planet detections will contribute
at least ten percent of the total blend flux. The separation of unassociated blended stars (neither the lens or
source star) from potential host flux will consider some
thought, but could be constrained through priors from
the event, such as the distance to the lens system.


5.5. _Limb darkening and wave optics_


We did not account for the effects of limb darkening
in our simulations (Witt 1995; Heyrovsk´y 2003). The



limb darkening profile of source stars is wavelength dependent, with the amplitude of the surface-to-limb variation decreasing _∝_ _λ_ _[−]_ [1] for the Sun (Hestroffer & Magnan 1998). Because the primary observations are in the
near infrared, _Roman_ typical source stars will exhibit
less limb darkening than in optical surveys. As shown
by Lee et al. (2009) and many more, the limb darkening profile alters the shape of the light curve (see their
Figure 6). This would likely impact our yield estimates
for low mass lenses most, where finite-source effects are
most likely. In our detection cuts, if a source is fainter
in its limb, it may shorten the effective timescale of the
event. This could lower _n_ 3 _σ_ or ∆ _χ_ [2] of the event in our
detection threshold. However, limb darkening increases
the peak magnification of events (see lower panel of Figure 2 which could modestly increase the number of detections we predict. We must also consider that _W146_
is a wide band and thus the limb darkening will have
a significant chromatic dependence over the wavelength
range of the filter (Han et al. 2000; Heyrovsk´y 2003;
Claret & Bloemen 2011).
If the mass of a lens is small enough ( _∼_ 10 _[−]_ [5] _M_ _⊕_ ),
the geometric optics description of microlensing becomes insufficient and wave effects manifest themselves
in the magnification curve (Takahashi & Nakamura
2003, among others). In short, the threshold for this
effect is when the wavelength of light being observed
becomes comparable to the Schwarzschild radius of the
lens; in this limit there is a fundamental limit to the
peak magnification of the event. For a mass of 10 _[−]_ [3] _M_ _⊕_
(3 _×_ 10 _[−]_ [9] _M_ _⊙_ ) this corresponds to wavelength of _∼_ 11 _µm_
(see Equations (5) and (7) of Sugiyama et al. (2020)).
This is below the long wavelength edge of the _W146_
band, and the corresponding wavelength only gets longer
for larger mass lenses. We therefore do not consider this
effect here.


5.6. _Mass Measurements_


The conversion from timescale to mass for FFP events

requires measurements of both the microlensing parallax and the angular Einstein ring. A measurement of
_ρ_ from finite-source effects and _θ_ _∗_ from the dereddened
source flux and colors(Yoo et al. 2004) would yield _θ_ E, if
the degeneracy discussed above can be broken and mea

_Roman_ FFPs 19



surements are made in another filter(s) while the source
is magnified [5] .
_Spitzer_ enabled the regular measurement of microlensing parallaxes to a large number of stellar, binary, and
bound-planetary microlensing events by levering the fact
that it was separated by the Earth by _∼_ au due to its
Earth-trailing orbit (Gould 1994, 1995, 1999), but these
events had projected Einstein ring sizes of a few au (e.g.,
Dong et al. 2007; Yee et al. 2015). Zhu & Gould (2016)
quantify the potential for simultaneous ground-based
observations (and _Roman_ -only observations) to measure
one- and two-dimensional microlens parallaxes. Spacebased parallax measurements of FFP lenses was also attempted using the Kepler spacecraft during the K2 Campaign 9 survey, which largely consisted of a microlensing survey toward the bulge (Henderson & Shvartzvald
2016; Henderson et al. 2016; Penny et al. 2017; Zhu
et al. 2017a,b; Zang et al. 2018). Penny et al. (2019)
and Bachelet & Penny (2019) show that the short intraL2 baseline between the _Euclid_ and _Roman_ spacecraft
would be enough to measure free-floating planet parallaxes. Ban (2020) computes probabilities for measuring parallaxes for combinations of ground and space
based telescopes. Concurrent observations with widefield infrared observatories, such as UKIRT (Hodapp et
al. 2018), VISTA (Dalton et al. 2006), and PRIME (Yee
et al. 2018) [6], as well as wide-field optical observatories,
such as DECam (Flaugher et al. 2015), HyperSuprimeCam (Miyazaki et al. 2012), and the Vera C. Rubin
Observatory (LSST Science Collaboration et al. 2009),
would enable parallax measurements for both bound and
free floating planets.


6. CONCLUSION


We have used GULLS simulation software (Penny et
al. 2019) to show that _Roman Galactic Exoplanet Survey_
will inform our understanding of the isolated compact
object mass function throughout the Galaxy, down to
very low planetary-mass objects. In particular, it will be
able detect microlensing events with timescales as short
as 1.5 hr, and thus isolated lenses with masses down to
at least 0.1 _M_ _⊕_ . This data set will be used to address



questions about both the low mass tail of the initial
mass function of stars as well as the total mass and

mass function of objects ejected from planetary systems
during planet formation and evolution. _Roman_ will be
able to probe populations of free-floating planets that
are essentially impossible to access from ground-based
microlensing surveys. Finally, the limits that _Roman_
will place if no such objects are detected would be the
most stringent to date by orders of magnitude.


ACKNOWLEDGMENTS


We are particularly proud to honor Nancy Grace Roman, after whom this survey telescope has recently been
named. We hope to live up to her extraordinary influence on space astronomy.


We appreciate the revisions from the referee that improved the quality of this work, as well as those from
careful readings by Radek Poleski and Przemek Mr´oz.
We thank our colleagues Andrew Gould and David Bennett for useful discussions. We thank everyone on the
_Roman_ Galactic Exoplanet Survey Science Investigation
Team. We also appreciate Exoplanet Lunch at Ohio
State University which was the source of many useful
discussions. SAJ dedicates his contribution to this work

to David John Prahl Will, who without this work and
that of many others would not be possible.
This work was performed in part under contract with
the California Institute of Technology (Caltech)/Jet
Propulsion Laboratory (JPL) funded by NASA through
the Sagan Fellowship Program executed by the NASA
Exoplanet Science Institute. S.A.J, M.T.P., and B.S.G.
were supported by NASA grant NNG16PJ32C and the
Thomas Jefferson Chair for Discovery and Space Exploration.


_Software:_ astropy (Astropy Collaboration et al.
2013, 2018), Matplotlib (Hunter et al. 2007), MulensModel (Poleski & Yee 2018), VBBinaryLensing (Bozza
2010; Bozza et al. 2018)



REFERENCES


Adams, F. C., Anderson, K. R., & Bloch, A. M. 2013,



5 We note that typically the empirical relations used to convert
from source flux and color to angular radius are based on measurements from giant stars, which are most likely to exhibit finitesource effects in microlensing events (e.g. van Belle 1999). For
lenses with low enough masses, however, we will need appropriately calibrated relations for non-giant source stars (e.g. Adams
et al. 2018).



MNRAS, 432, 438


6 https://www.astro.caltech.edu/ _∼_ srk/ _\_ Workshops/TDAMMS/
[Files4Facilities/PRIME.pdf](https://www.astro.caltech.edu/~srk/\Workshops/TDAMMS/Files4Facilities/PRIME.pdf)


20 Johnson et al.



Adams, A. D., Boyajian, T. S., & von Braun, K. 2018,

MNRAS, 473, 3608


Alcock, C., Allsman, R. A., Alves, D., et al. 1996, ApJ, 471,

774


Alcock, C., Allsman, R. A., Alves, D., et al. 1997, ApJ, 486,

697


Alcock, C., Allsman, R. A., Alves, D., et al. 1998, ApJL,

499, L9


Agol, E. 2003, ApJ, 594, 449


Akeson, R., Armus, L., Bachelet, E., et al. 2019, arXiv

e-prints, arXiv:1902.05569


Astropy Collaboration, Robitaille, T. P., Tollerud, E. J., et

al. 2013, A&A, 558, A33


Astropy Collaboration, Price-Whelan, A. M., Sip˝ocz, B. M.,

et al. 2018, AJ, 156, 123


Bachelet, E., & Penny, M. 2019, ApJL, 880, L32


Bailey, V. P., Armus, L., Balasubramanian, B., et al. 2019,

arXiv e-prints, arXiv:1901.04050


Ban, M., Kerins, E., & Robin, A. C. 2016, A&A, 595, A53


Ban, M. 2020, arXiv e-prints, arXiv:2003.04504


Barclay, T., Quintana, E. V., Raymond, S. N., & Penny,

M. T. 2017, ApJ, 841, 86


Bardalez Gagliuffi, D. C., Burgasser, A. J., Schmidt, S. J.,

et al. 2019, ApJ, 883, 205


Bate, M. R. 2009, MNRAS, 392, 590


Batygin, K., & Brown, M. E. 2016, AJ, 151, 22


Bennett, D. P., & Rhie, S. H. 2002, ApJ, 574, 985


Bennett, D. P., Sumi, T., Bond, I. A., et al. 2012, ApJ, 757,

119


Bonnell, I. A., Clark, P., & Bate, M. R. 2008, MNRAS,

389, 1556


Borucki, W. J., Koch, D., Basri, G., et al. 2010, Science,

327, 977


Bozza, V. 2010, MNRAS, 408, 2188


Bozza, V., Bachelet, E., Bartoli´c, F., et al. 2018, MNRAS,

479, 5157


Cassan, A., Kubas, D., Beaulieu, J.-P., et al. 2012, Nature,

481, 167


Chung, S.-J., Zhu, W., Udalski, A., et al. 2017, ApJ, 838,

154


Clanton, C., & Gaudi, B. S. 2017, ApJ, 834, 46


Claret, A., & Bloemen, S. 2011, A&A, 529, A75


Collier Cameron, A., Guenther, E., Smalley, B., et al. 2010,

MNRAS, 407, 507


Cushing, M. C., Kirkpatrick, J. D., Gelino, C. R., et al.

2011, ApJ, 743, 50


Dalton, G. B., Caldwell, M., Ward, A. K., et al. 2006,

Proc. SPIE, 62690X



Debes, J. H., Ygouf, M., Choquet, E., et al. 2016, Journal

of Astronomical Telescopes, Instruments, and Systems, 2,

011010

Di Stefano, R., & Scalzo, R. A. 1999, ApJ, 512, 564

Dong, S., Udalski, A., Gould, A., et al. 2007, ApJ, 664, 862

Doyle, L. R., Carter, J. A., Fabrycky, D. C., et al. 2011,

Science, 333, 1602

Flaugher, B., Diehl, H. T., Honscheid, K., et al. 2015, AJ,

150, 150

Gagn´e, J., Faherty, J. K., Mamajek, E. E., et al. 2017,

ApJS, 228, 18

Gaudi, B. S. 2012, ARA&A, 50, 411

Gaudi, B. S., Stassun, K. G., Collins, K. A., et al. 2017,

Nature, 546, 514

Gaudi, B. S., Akeson, R., Anderson, J., et al. 2019, BAAS,

51, 211

Gillon, M., Triaud, A. H. M. J., Demory, B.-O., et al. 2017,

Nature, 542, 456.

Godines, D., Bachelet, E., Narayan, G., et al. 2019,

Astronomy and Computing, 28, 100298

Gould, A., & Loeb, A. 1992, ApJ, 396, 104

Gould, A. 1994, ApJL, 421, L75

Gould, A. 1995, ApJL, 441, L21

Gould, A., & Welch, D. L. 1996, ApJ, 464, 212

Gould, A. 1997, Variables Stars and the Astrophysical

Returns of the Microlensing Surveys, 125

Gould, A., & Gaucherel, C. 1997, ApJ, 477, 580

Gould, A. 1999, ApJ, 514, 869

Gould, A., Gaudi, B. S., & Han, C. 2003, ApJL, 591, L53

Griest, K. 1991, ApJ, 366, 412

Griest, K., Cieplak, A. M., & Lehner, M. J. 2014, ApJ, 786,

158

Hawley, S. L., Davenport, J. R. A., Kowalski, A. F., et al.

2014, ApJ, 797, 121

Han, C., & Kang, Y. W. 2003, ApJ, 596, 1320

Han, C., Park, S.-H., & Jeong, J.-H. 2000, MNRAS, 316, 97

Han, C., Chung, S.-J., Kim, D., et al. 2004, ApJ, 604, 372

Han, C., Gaudi, B. S., An, J. H., & Gould, A. 2005, ApJ,

618, 962

Han, C., Udalsk, A., Gould, A., et al. 2020, AJ, 159, 91

Han, C., Lee, C.-U., Udalski, A., et al. 2020, AJ, 159, 134

Henderson, C. B., & Shvartzvald, Y. 2016, AJ, 152, 96

Henderson, C. B., Poleski, R., Penny, M., et al. 2016,

PASP, 128, 124401

Heyrovsk´y, D. 2003, ApJ, 594, 464

Hestroffer, D., & Magnan, C. 1998, A&A, 333, 338

Hong, Y.-C., Raymond, S. N., Nicholson, P. D., & Lunine,

J. I. 2018, ApJ, 852, 85

Hodapp, K. W., Kerr, T., Varricatt, W., et al. 2018,

Proc. SPIE, 107002Z


_Roman_ FFPs 21



Hounsell, R., Scolnic, D., Foley, R. J., et al. 2018, ApJ, 867,

23

Hunter, J. D. 2007, CISE, 9, 3

Kerins, E., Robin, A. C., & Marshall, D. J. 2009, MNRAS,

396, 1202

Khakpash, S., Penny, M., & Pepper, J. 2019, AJ, 158, 9

Kim, S.-L., Lee, C.-U., Park, B.-G., et al. 2016, Journal of

Korean Astronomical Society, 49, 37

Kirkpatrick, J. D., Gelino, C. R., Cushing, M. C., et al.

2012, ApJ, 753, 156

Kreidberg, L., Luger, R., & Bedell, M. 2019, ApJL, 877,

L15

Lee, C.-H., Riffeser, A., Seitz, S., et al. 2009, ApJ, 695, 200.

L´eger, A., Rouan, D., Schneider, J., et al. 2009, A&A, 506,

287

Liebes, S. 1964, Physical Review, 133, 835

Luhman, K. L. 2012, ARA&A, 50, 65

LSST Science Collaboration, Abell, P. A., Allison, J., et al.

2009, arXiv e-prints, arXiv:0912.0201

Ma, S., Mao, S., Ida, S., et al. 2016, MNRAS, 461, L107

Malmberg, D., Davies, M. B., & Heggie, D. C. 2011,

MNRAS, 411, 859

Mao, S., & Paczynski, B. 1991, ApJL, 374, L37

Mao, S., & Paczynski, B. 1996, ApJ, 473, 57

Miyazaki, S., Komiyama, Y., Nakaya, H., et al. 2012,

Proc. SPIE, 84460Z

Moniez, M. 2010, General Relativity and Gravitation, 42,

2047

Montet, B. T., Yee, J. C., & Penny, M. T. 2017, PASP, 129,

044401

Mr´oz, P., Udalski, A., Skowron, J., et al. 2017, Nature, 548,

183

Mr´oz, P., Ryu, Y.-H., Skowron, J., et al. 2018, AJ, 155, 121

Mr´oz, P., Udalski, A., et al. 2019, A&A, 622, A201

Mr´oz, P., Udalski, A., Skowron, J., et al. 2019, ApJS, 244,

29

Mr´oz, P., Poleski, R., Han, C., et al. 2020, AJ, 159, 262

Muraki, Y., Sumi, T., Abe, F., et al. 1999, Progress of

Theoretical Physics Supplement, 133, 233

Niikura, H., Takada, M., Yasuda, N., et al. 2019, Nature

Astronomy, 3, 524

Niikura, H., Takada, M., Yokoyama, S., et al. 2019, PhRvD,

99, 083503

Nielsen, E. L., De Rosa, R. J., Macintosh, B., et al. 2019,

AJ, 158, 13

Mayor, M., & Queloz, D. 1995, Nature, 378, 355

Paczy´nski, B. 1991, ApJL, 371, L63

Paczy´nski, B. 1986, ApJ, 304, 1

Penny, M. T., Kerins, E., Rattenbury, N., et al. 2013,

MNRAS, 434, 2



Penny, M. T., Rattenbury, N. J., Gaudi, B. S., & Kerins, E.

2017, AJ, 153, 161

Penny, M. T., Gaudi, B. S., Kerins, E., et al. 2019, ApJS,

241, 3

Penny, M., Bachelet, E., Johnson, S., et al. 2019, BAAS,

51, 563

Poleski, R., & Yee, J. 2018, MulensModel: Microlensing

light curves modeling, ascl:1803.006

Pfyffer, S., Alibert, Y., Benz, W., et al. 2015, A&A, 579,

A37

Rasio, F. A., & Ford, E. B. 1996, Science, 274, 954

Renault, C., Afonso, C., Aubourg, E., et al. 1997, A&A,

324, L69

Robin, A. C., Reyl´e, C., Derri`ere, S., & Picaud, S. 2003,

A&A, 409, 523

Robin, A. C., Marshall, D. J., Schultheis, M., & Reyl´e, C.

2012, A&A, 538, A106

Sajadian, S., & Poleski, R. 2019, ApJ, 871, 205

Shvartzvald, Y., Bryden, G., Gould, A., et al. 2017, AJ,

153, 61

Skowron, J., Udalski, A., Gould, A., et al. 2011, ApJ, 738,

87

Specht, D., Kerins, E., Awiphan, S., et al. 2020, arXiv

e-prints, arXiv:2005.14668

Spergel, D., Gehrels, N., Baltay, C., et al. 2015, arXiv

e-prints, arXiv:1503.03757.

Spiegel, D. S., & Burrows, A. 2012, ApJ, 745, 174

Strigari, L. E., Barnab`e, M., Marshall, P. J., et al. 2012,

MNRAS, 423, 1856

Sugiyama, S., Kurita, T., & Takada, M. 2020, MNRAS,

493, 3632

Sumi, T., Wo´zniak, P. R., Udalski, A., et al. 2006, ApJ,

636, 240

Sumi, T., Kamiya, K., Bennett, D. P., et al. 2011, Nature,

473, 349

Sumi, T., Bennett, D. P., Bond, I. A., et al. 2013, ApJ, 778,

150

Sumi, T., & Penny, M. T. 2016, ApJ, 827, 139

Suzuki, D., Bennett, D. P., Sumi, T., et al. 2016, ApJ, 833,

145

Takahashi, R., & Nakamura, T. 2003, ApJ, 595, 1039

Teachey, A., & Kipping, D. M. 2018, Science Advances, 4,

eaav1784

Teachey, A., Kipping, D., Burke, C. J., et al. 2020, AJ, 159,

142

Terry, S. K., Barry, R. K., Bennett, D. P., et al. 2020, ApJ,

889, 126

Troxel, M. A., Long, H., Hirata, C. M., et al. 2019, arXiv

e-prints, arXiv:1912.09481

Trujillo, C. A., & Sheppard, S. S. 2014, Nature, 507, 471


22 Johnson et al.



Tsiganis, K., Gomes, R., Morbidelli, A., et al. 2005, Nature,

435, 459

Udalski, A., Szymanski, M., Kaluzny, J., et al. 1992, AcA,

42, 253

van Belle, G. T. 1999, PASP, 111, 1515

van Elteren, A., Portegies Zwart, S., Pelupessy, I., et al.

2019, A&A, 624, A120

Veras, D., Wyatt, M. C., Mustill, A. J., et al. 2011,

MNRAS, 417, 2104

Vietri, M., & Ostriker, J. P. 1983, ApJ, 267, 488

Witt, H. J. 1995, ApJ, 449, 42

Witt, H. J., & Mao, S. 1994, ApJ, 430, 505

Wolszczan, A., & Frail, D. A. 1992, Nature, 355, 145

Wo´zniak, P., & Paczy´nski, B. 1997, ApJ, 487, 55

Wright, E. L., Eisenhardt, P. R. M., Mainzer, A. K., et al.

2010, AJ, 140, 1868



Wyrzykowski, �L., Rynkiewicz, A. E., Skowron, J., et al.

2015, ApJS, 216, 12

Yee, J. C., Udalski, A., Calchi Novati, S., et al. 2015, ApJ,

802, 76

Yee, J. C., Anderson, J., Akeson, R., et al. 2018, arXiv

e-prints, arXiv:1803.07921

Yoo, J., DePoy, D. L., Gal-Yam, A., et al. 2004, ApJ, 603,

139

Zang, W., Penny, M. T., Zhu, W., et al. 2018, PASP, 130,

104401

Zhu, W. & Gould, A. 2016, Journal of Korean

Astronomical Society, 49, 93

Zhu, W., Udalski, A., Huang, C. X., et al. 2017, ApJL, 849,

L31

Zhu, W., Huang, C. X., Udalski, A., et al. 2017, PASP, 129,

104501


_Roman_ FFPs 23


APPENDIX



A. AN INTRODUCTION TO MICROLENSING IN

THE LARGE ANGULAR SOURCE REGIME.


A.1. _Light curve morphology_


For a typical isolated microlens, the angular size of
the source is much smaller than the angular size of the
Einstein ring of the lens, and thus the approximation
of a point-source generally remains valid. That is, the
magnification as a function of the separation between
the source and the lens normalized to the size of the
Einstein ring _u_ is given by (Paczy´nski 1986),


_u_ [2] + 2
_A_ = _u_ ( _u_ [2] + 4) [1] _[/]_ [2] _[.]_ (A1)


The magnification peaks at the minimum separation
_u_ 0 = _θ_ 0 _/θ_ E where _θ_ 0 is the angular separation between
the source and the lens at closest approach. The point
source approximation in A.1 breaks down when angular separation of the source from the lens, _θ_ 0, becomes
comparable to the angular radius of the source star _θ_ _∗_ .
For point lenses, this condition results in a significant
second derivative in the point lens light curve over the
angular size of the source, which must be accounted for
computing the magnification. Thus, for events with impact parameter such that _ρ/u_ 0 ≳ 0 _._ 5, where _ρ_ = _θ_ _∗_ _/θ_ E,
the peak of the event (at times _|t −_ _t_ 0 _|/t_ E ≲ 2 _ρ_ ) is affected by finite-source effects (Gould & Gaucherel 1997).
Since, for stellar mass lenses, _ρ_ is typically in the range
of 10 _[−]_ [3] _−_ 10 _[−]_ [2], most events are unaffected, and those
that are effected are high-magnification events. Even in
such events, finite-source effects are only detectable near
the peak of the event, while the magnification during the
rest of the event is essentially equivalent to that due to
a point source.
However, this characterization breaks completely
when the angular size of the source becomes comparable to the angular size of the Einstein ring, or _ρ_ ≳ 1. In
particular, in the extreme case when _ρ ≫_ 1, the source
will completely envelop the Einstein ring of the lens if it
passes within the angular source radius. When this happens, the lens magnifies only a fraction of the area of the
source as it transits the disk of the source (e.g., Gould
& Gaucherel 1997; Agol 2003). In this limit ( _ρ ≫_ 1) and
to first order, the magnification curve can take on a ‘top
hat’ or boxcar shape, saturating at a magnification of

_A_ peak _≈_ 1 + [2] (A2)

_ρ_ [2]


(Liebes 1964; Gould & Gaucherel 1997; Agol 2003). The
light curve shape is essentially independent of _u_ 0 except



when _u_ 0 _∼_ _ρ_ (Agol 2003). Furthermore, the duration of
the event is no longer set by the microlensing timescale
_t_ E, but rather is proportional to the source radius crossing time _t_ _∗_ = _θ_ _∗_ _/µ_ rel = _t_ E _ρ_ . Mr´oz et al. (2018a) account
for the impact parameter _u_ 0 such that they use the time
taken for the lens to cross the chord of the source



where _u_ 0 _,∗_ = _θ_ 0 _/θ_ _∗_ . Note that this timescale is independent of the angular radius of the Einstein ring and thus
lens mass. We note that Mr´oz et al. (2017) define _t_ _∗_ as
the crossing time over the chord of the source, but we
define _t_ _∗_ is the source radius crossing time defined above
(see Appendix A of Skowron et al. 2011). Henceforth,
we will use _t_ _c_ for the chord crossing time and we propose
that this become the convention.

Broadly, these changes in light curve morphology are
referred to as extreme finite-source effects, as the light
curve is affected by finite-source effects throughout the
duration (e.g., at no time while the source is magnified
does the point source approximation hold). We demonstrate the impact of FSEs on the light curve of events
in Figure 12. We consider five lenses in which we only
vary the angular size of the Einstein ring, quantified by
_ρ_ = 0 _._ 25, 0.50, 1.00, 2.00, and 4.00 (as the size of the
source is fixed). The sizes of these rings are shown in the
upper left corner of the leftmost panel scaled to the size
of a source star, which is depicted as a gray circle (with
an angular radius of _θ_ _∗_ ). A horizontal, gray line depicts
the path the lenses will take, and it is separated from
the center of the source star by the impact parameter _θ_ _∗_
(again, _to scale_ ). We then vary the impact parameter
from _u_ 0 _,∗_ = 3 _._ 0 down to _u_ 0 _,∗_ = 0 _._ 0 from left to right.
This is written above the plot and depicted as the source
star (gray circle) approaching the lens trajectory (gray
line). Note that the circles and their separations are
independent of the time and magnification axes.
Each panel depicts all five lenses in different events,
where the line-style matches the lens with the same Einstein ring line-style in the upper left corner of the leftmost panel. We use the method of Lee et al. (2009) as
implemented in `MulensModel` (Poleski & Yee 2018) to
compute all light curves in this figure.
As _θ_ E _∝_ _√M_, the more massive lens will have the

largest _θ_ E and be the furthest from the regime of finitesource effects. Note that time is referenced to the peak
of the event ( _t_ 0 ), and scaled by the analytic timescale _t_ E
on the horizontal axis. The solid-line light curve behaves



_t_ _c_ = [2] _[θ]_ _[∗]_

_µ_ rel



~~�~~ 1 _−_ ( _u_ 0 _,∗_ ) [2] _,_ (A3)


24 Johnson et al.




















|θ0/θ ∗= 3.00|θ0/θ ∗= 1.50|θ0/θ ∗= 0.75|θ0/θ ∗= 0.00|Col5|
|---|---|---|---|---|
|Lens Trajectory<br>|||||
|θ ∗<br>θ0<br>ρ = ~~θ~~<br>ρ<br>~~ρ~~<br>ρ<br>~~ρ~~<br>ρ|~~∗~~/θE<br> = 0.25<br>~~ =~~ 0.50<br> = 1.00<br>~~ =~~ 2.00<br> = 4.00||||
|θ ∗<br>θ0<br>ρ = ~~θ~~<br>ρ<br>~~ρ~~<br>ρ<br>~~ρ~~<br>ρ|~~∗~~/θE<br> = 0.25<br>~~ =~~ 0.50<br> = 1.00<br>~~ =~~ 2.00<br> = 4.00|||~~-~~<br>1|
|θ ∗<br>θ0<br>ρ = ~~θ~~<br>ρ<br>~~ρ~~<br>ρ<br>~~ρ~~<br>ρ|~~∗~~/θE<br> = 0.25<br>~~ =~~ 0.50<br> = 1.00<br>~~ =~~ 2.00<br> = 4.00|||~~-~~0<br>~~-~~0.<br>1|



**Figure 12.** The morphology of microlensing light curves changes as finite-source effects become more prominent. In the
background, we show a gray circle that represents the source (with an angular radius _θ_ _∗_ ). We also show five Einstein rings
scaled to the source size, which have _ρ_ = _θ_ _∗_ _/θ_ E values indicated in the legend. Each panel is for a different impact parameter
_u_ 0 _,∗_ = 3 _._ 00, 1.50, 1.00, and 0.00 from left to right. We change the scaled position of the source star circle relative to the lens
trajectory (gray horizontal line) to match the impact parameter. For each panel, we plot the magnification as a function of time
scaled to the microlensing timescale ( _t_ _−_ _t_ 0 ) _/t_ E . For the most extreme case of _ρ_ = 4 _._ 00, we see no appreciable magnification until
the lens traverses the source ( _u_ 0 _,∗_ _<_ 1) at which point the magnification is essentially constant (except when the lens is near the
edges of the source). The light curves thus have a ‘top hat’ appearance. We note that this ‘top hat’ morphology only appears
when there is no limb darkening. All events have peak magnifications that monotonically increase _u_ _∗_ decreases, however, this
maximum magnification begins to saturate at the expected value of 1 + _ρ_ 2 [2] [for] _[ ρ >]_ [ 1. However, the length of those events with]
_ρ >_ 1 are significantly longer than expected from their analytic _t_ E timescales.



essentially how you expect an isolated lens to behave as
the impact parameter drops up until the last two panels
where the peak becomes more rounded. This is the first
breakdown we described that occurs when _ρ/u_ 0 ≳ 0 _._ 5,
or when the size of the source star is within a few times

the impact parameter.
However, the behavior is dramatically different for the
lowest mass lens ( _ρ ≫_ 1). In this case, when lens is
not transiting the source ( _u_ _∗_ _>_ 1), there is effectively
no magnification. For the smallest impact parameter
(rightmost panel), the lightcurve looks like the top-hat
described earlier, magnifying the source by roughly 10%
(1 + 4 [2] [2] _[≈]_ [1] _[.]_ [13). Also note the duration of this event is]

now much longer than one would expect given its analytic timescale _t_ E . In fact when _u_ 0 _,∗_ = 0 _._ 00, the duration
is nearly exactly what we predict given the diameter
crossing time _t_ _c_ _/t_ E = 2 _ρ_ = 4 for the _ρ_ = 2 case and
_t_ _c_ _/t_ E = 8 for the _ρ_ = 4 case. In the rightmost panel,
the _ρ_ = 4 event lasts _∼_ 4 times longer than one would
expect based on the value of _t_ E .
To provide a quantitative sense of the relevant scales,
consider a typical stellar mass lens (0 _._ 3 _M_ _⊙_ ), which has
an angular Einstein ring radius of _θ_ E = 550 _µ_ as. A
source star in the Galactic bulge (at a distance of _D_ S = 8
kpc) that has a radius of 1 _R_ _⊙_ will have an angular
radius of just 0 _._ 6 _µ_ as. Lenses with mass ≲ 0 _._ 12 _M_ _⊕_ will



have _ρ_ ≳ 1 for this source. A typical clump giant in the
bulge will have a radius of _∼_ 10 _R_ _⊙_, leading to _ρ >_ 1 for
lenses with mass ≲ 10 _M_ _⊕_ .
These morphological changes will impact the microlensing event rate and microlensing optical depth (Vietri & Ostriker 1983; Paczy´nski 1991). Recall that the
microlensing optical depth (the probability any given
star is being lensed) is a function of the fraction of the
sky covered by Einstein rings. As demonstrated above,
lenses with small enough masses will have Einstein rings
smaller than the angular size of some stars. Han et al.
(2005) show that for lenses with low enough masses,
the event rate actually _increases_ compared to what you
would expect for lenses with Einstein rings smaller than
the angular size of source stars. For these lenses, the
event rate is proportional to the fraction of the sky covered by source stars. However, the detection of such
events is hampered by the fact that the peak magnification is lower than one would expect for a point source.
Han et al. (2005) also derive analytic expressions for the
threshold impact parameter for detection and the minimum detectable mass lens as function of the threshold

signal-to-noise ratio for detection.
In reality, the shape of the light curve is sensitive to
the limb-darkening profile of the source as well as any
of its surface features (e.g. Witt & Mao 1994; Gould


_Roman_ FFPs 25



& Welch 1996; Agol 2003; Heyrovsk´y 2003; Yoo et al.
2004; Lee et al. 2009). The impact of included limb
darkening is shown in the lower panel of Figure 2. The
‘top hat’ shape disappears, and the light curve becomes
more rounded. The example in Figure 2 adopted a single parameter linear limb darkening profile, but more
structure could be added if a more complex profile was
used (e.g., Claret & Bloemen 2011), or if surface features
(such as star spots) were considered (Heyrovsk´y 2003).


B. DETECTION CRITERIA


We require that simulated events pass two criteria to
qualify as detections. The first is based on the deviation
(∆ _χ_ [2] ) the event causes from a flat light curve


∆ _χ_ [2] = _χ_ [2] Line _[−]_ _[χ]_ [2] FSPL (B4)


where _χ_ [2] Line [is the] _[ χ]_ [2] [ value of the simulated light curve]
for a flat line at the baseline flux and _χ_ [2] FSPL [is the same]
but for the simulated data to the injected finite-source
point-lens model of the event. The second criterion is
the number of consecutive data points that are measured
3 _σ_ . In this section we isolate the effect of the value of
each criterion on the yield as a function of planet mass in
turn, and then consider the complex interplay between
them.
We first plot the cumulative number of detected events
_N_ det ( _X ≥_ ∆ _χ_ [2] ) as a function of the threshold ∆ _χ_ [2] in
Figure 13. We show the cases of our five discrete masses
under the assumption that there are one such planet per
star. Applying only this criterion, we can analytically
estimate the impact of mission/survey design changes on
the yield of FFPs by inferring the impact those changes
would have on the ∆ _χ_ [2] of events (akin to Paper I). This
is because the distributions in Figure 13 can be locally
fit by a power law


_N_ (∆ _χ_ [2] _> X_ ) _∝_ _X_ _[α]_ _,_ (B5)


over a wide range of ∆ _χ_ [2], as has previously been shown
by Bennett & Rhie (2002). We fit this power law for
each mass on the range ∆ _χ_ [2] = [150 _,_ 600], and list the
values of exponent in Table 4.
While we necessarily employ a ∆ _χ_ [2] _≥_ 300 as one of
our thresholds, basing detection rates solely on this criterion is problematic because of the potential for very
short events, e.g., events with only a few extremely magnified points that together contribute more than 300
to the ∆ _χ_ [2] . Modeling these events would be challenging. We therefore also impose the second criterion on
_n_ 3 _σ_, which is specifically the number of consecutive data
points with _n >_ 3 _σ_ above the baseline flux of the source





















**Figure 13.** The cumulative distribution of the ∆ _χ_ [2] of
simulated events. From bottom (red) to top (purple), the
solid lines represent yields of lenses with masses of 0 _._ 1 _M_ _⊕_ to
10 [3] _M_ _⊕_, assuming one FFP of that mass per star in the MW.
From left to right, the vertical, black dashed lines indicate
where ∆ _χ_ [2] = 150, ∆ _χ_ [2] = 300, and ∆ _χ_ [2] = 600. We fit a
power law (Eqn. B5) to each of the solid lines for a range of
∆ _χ_ [2] _∈_ [150 _,_ 600]. The slopes of these are included in Table
4. We also plot the cumulative ∆ _χ_ [2] distributions for events
with _n_ 3 _σ_ _≥_ 3 and _≥_ 6 as the long-dashed and dashed lines,
respectively. The distributions flatten significantly for lower
∆ _χ_ [2] when we require _n_ 3 _σ_ _≥_ 6.


star plus blend [7] . This criterion ensures that there will
be a sufficient number of high signal-to-noise ratio data
points during the events that they can be confidently
modelled.

To illustrate how this criterion changes the cumulative number of detections relative to just applying the
∆ _χ_ [2] criterion, in Figure 13 we plot as dashed (longdashed) lines the distributions also requiring _n_ 3 _σ_ _≥_ 6
( _n_ 3 _σ_ _≥_ 3) points. We fit the slopes of the cumulative
distributions as power laws distributions as before over
the same range, and include the power law exponents in
Table 4. For a given mass, the distributions we derive
applying both criteria change relative to only applying
the ∆ _χ_ [2] criterion in a manner that depend on the mass
of the lens.

We note that, for all of the masses, the cumulative
distributions begin to fall below the power law fit to the
solid curves (without the _n_ 3 _σ_ cut) at the highest values
of the threshold ∆ _χ_ [2] . Furthermore, the onset of this
deviation occurs for lower values of ∆ _χ_ [2] for the very
smallest masses. This deviation is due to the onset of


7 We note that a similar criterion were used by Sumi et al. (2011)
and Mr´oz et al. (2017)


26 Johnson et al.



**Table 4.** Slopes of ∆ _χ_ [2] distributions


( _M_ _⊕_ ) _α_


_n_ 3 _σ_ _≥_ 0 _n_ 3 _σ_ _≥_ 3 _n_ 3 _σ_ _≥_ 6


0.1 _−_ 0 _._ 36 _−_ 0 _._ 28 _−_ 0 _._ 15

1 _−_ 0 _._ 21 _−_ 0 _._ 19 _−_ 0 _._ 12

10 _−_ 0 _._ 18 _−_ 0 _._ 17 _−_ 0 _._ 13

100 _−_ 0 _._ 14 _−_ 0 _._ 080 _−_ 0 _._ 040

1000 _−_ 0 _._ 097 _−_ 0 _._ 013 _−_ 0 _._ 0020


finite-source effects, and the increasing importance of
these effects for lower masses.
Conversely, for lower values of the threshold ∆ _χ_ [2], the
cumulative distributions begin to fall below the power
law fit to the solid curves at roughly the same value of
∆ _χ_ [2] for the three largest masses, but at different values
for the lowest two masses. Finally, for all the masses,
the cumulative distribution of the number of detections
becomes essentially flat (independent of the ∆ _χ_ [2] threshold) for values of ∆ _χ_ [2] ≲ 150 and _n_ 3 _σ_ _≥_ 6. Thus, for
this combination of detection criteria, the yield does not
improve with a lower ∆ _χ_ [2] threshold, only with changing the _n_ 3 _σ_ cut. These behaviors are all consistent with
expectations based on the gradual change in the morphologies of the light curves as finite-source effects begin
to dominate (roughly for masses ≲ _M_ _⊕_ (See Section A).
To further explore the interplay between the two detection criteria, we isolate the effect of the _n_ 3 _σ_ cut on
the yields in Figure 14. Here we show the cumulative
fraction of events as a function of _n_ 3 _σ_ for events with
∆ _χ_ [2] _≥_ 300.
For the two largest masses, the yield is a relatively
weak function of _n_ 3 _σ_ since these masses typically give
rise to longer timescale (and thus more well sampled)
events. Interestingly, we find that 10 _M_ _⊕_ events are the
most robust to this selection criterion for _n_ 3 _σ_ ≲ 10, however it falls off quickly afterwards, as expected. The lowest two masses continue this trend, become ever more
sensitive to the value of the _n_ 3 _σ_ cut at a fixed threshold
of ∆ _χ_ [2] _≥_ 300. Again, this is expected as the timescale
distributions for the lower and lower masses are typically shorter and shorter compared to the cadence of 15

minutes.
Thus we find that there an important and complex
interplay between both these criteria, which makes predicting the impact of changes in yield at different values
of the photometric precision at a given magnitude more
difficult than if we just imposed the ∆ _χ_ [2] threshold. As
a concrete example to illustrate this point, imagine an



0 _._ 3


0 _._ 2


0 _._ 10


**Figure 14.** The cumulative fraction of events as a function
of _n_ 3 _σ_ with ∆ _χ_ [2] _>_ 300. Each line represents only events
with the labeled mass. The left (right) vertical dashed lines
are at _n_ 3 _σ_ = 3 ( _n_ 3 _σ_ = 6). The most significant difference
between these thresholds is for the yields of 0 _._ 1 _M_ _⊕_ FFPs,
which nearly doubles when the threshold is relaxed. These
events are typically short or have low magnification. For
masses above 10 _M_ _⊕_, the number of detections is relatively
independent over the range 6 _< n_ 3 _σ_ _<_ 40 . We conclude
that the impact of our _n_ 3 _σ_ detection criterion is highest for
low-mass lensing events.


event with 5 data points 8 _σ_ above the baseline flux and
the next most significant point being only 3 _σ_ above baseline. Further assume that all these points are consecutive, and together yield a total ∆ _χ_ [2] = 329. We could
change our threshold to ∆ _χ_ [2] _≥_ 329, and our event would
still be counted as detected (at it still passes the _n_ 3 _σ_ _≥_ 6
cut).
As discussed previously, changing the threshold in
∆ _χ_ [2] is equivalent to scaling the photometric precision of
the survey as function of magnitude. However, simply
scaling the yield with the threshold ∆ _χ_ [2] doesn’t capture
the impact on the _n_ 3 _σ_ _≥_ 6. What is really of interest is
how the number of detected events when we rescale the

individual uncertainties including both criteria. This
demonstrates how robust the yield is to degradation or
improvement in the photometry. In the above example,
assume the uncertainties are increased by _∼_ 4 _._ 7%, such
that the event now has ∆ _χ_ [2] = 300 and thus would still
(barely) pass the ∆ _χ_ [2] criterion for detection. However,
the last point would now have a significance of _∼_ 2 _._ 9 _σ_,
and thus the event would fail our _n_ 3 _σ_ criterion, and no
longer be detected. This means that those distributions
in Figure 13 and 14 can only be used to predict the
change in the expected number of detections resulting
from changes in the two detection criteria, but cannot
be used to assess the impact on the yield if the photo


1 _._ 00

0 _._ 9
0 _._ 8
0 _._ 7

0 _._ 6

0 _._ 5


0 _._ 4






metric precision changes at fixed magnitude. The latter
is of more interest when estimating the changes in the
survey yield as a result of changes in the mission design.
Thus, to further investigate this interplay, we ran a
separate set of modified simulations where everything is
the same as described in Section 2, except that we added
two calculations. For every event, we calculated the factor by which the uncertainties would need to be uniformly scaled by in order that the total ∆ _χ_ [2] of the event
is equal to 300, specifically _C_ DC2 = (∆ _χ_ [2] True _[/]_ [300)] [1] _[/]_ [2] _[.]_ [ We]
then find the data point that would be the last to qualify
the event for _n_ 3 _σ_ _≥_ 6 cut, and calculate the factor that
the photometric uncertainty of that data point would
need to be scaled to reach a 3 _σ_ significance, _C_ N3S . The
lesser of these two factors is the more stringent criteria, and we can therefore assess how the yield changes
when the photometric uncertainty is changed including
the impact of both both criteria. We find the cumulative
distribution of detections as a function of the minimum

of these two scaling factors and call it the ‘Uncertainty
Scale Factor’= min( _C_ DC2 _, C_ N3S ). We plot this distribution normalized to the number of detections expected
when the error bars are not scaled at all for our five
reference masses in Figure 15.
As one would expect, the larger mass lenses (10 [3] and
10 [2] _M_ _⊕_ have the essentially the same dependence on this
scaling factor. These events typically last longer, so
the _n_ 3 _σ_ = 6 criterion is generally not approached. For
both these masses, the behavior is thus the same and
is similar to just scaling ∆ _χ_ [2] . As the mass of lenses
drops, the distributions begin to increasingly steepen
from 10 to 1 to 0.1 _M_ _⊕_ (the green dot-dash, orange long
dash, and red solid lines). As events become shorter
and shorter with decreasing mass, more events go undetected due to the consideration of _n_ 3 _σ_ when the uncertainties are increased. Fortunately, the inverse is also
true. In fact, a larger fraction of events are recovered
when the uncertainties are smaller than expected (the
spread is larger between the distributions for scale factors less than unity).
The most important takeaway from Figure 15 is that
the number of detections is fairly robust to the precise _Roman_ photometric uncertainties that are achieved
across a broad range in lens masses, and there are no
”thresholds” in the photometric precision below which
the detection rate drops precipitously. As a concrete example, for the mostly highly impacted lens mass 0 _._ 1 _M_ _⊕_,
if precision is 10% greater than expected we still recover
_∼_ 80% of events.



2.0

1.8

1.6


1.4


1.2


1.0


0.8


0.6







|27|Col2|Col3|
|---|---|---|
||||
||Fiducial Uncertainties|Fiducial Uncertainties|
|103_M⊕_<br>102_M⊕_<br>10_M⊕_<br>1_M⊕_<br>0_._1_M⊕_|||
|.50<br>0.75<br>0.9<br>1.0 1.1<br>1.25<br>Uncertainty Scale Factor|.50<br>0.75<br>0.9<br>1.0 1.1<br>1.25<br>Uncertainty Scale Factor|1.50|


**Figure 15.** The number of low-mass lens detections that
pass both our detection criteria has a higher dependence
on the _Roman_ photometric precision than high-mass lenses.
These distributions are normalized to the number of events in

each mass bin detected when no scaling is applied. The distributions for events with 10 [2] and 10 [3] _M_ _⊕_ behave as though
only the ∆ _χ_ [2] threshold were being scaled, as these events
have long timescales and thus are generally robust to the
_n_ 3 _σ_ threshold. However, as lens mass decreases, the slopes of
the curves for the lower masses events continues to steepen
for higher error scalings. These masses naturally produce
shorter timescale events, which are much more susceptible
to cuts in _n_ 3 _σ_ . Nevertheless, when considering our detection
criteria, these distributions show that _Roman_ ’s yield of lowmass lensing events will degrade gracefully with increasing
(fractional) photometric precision.


