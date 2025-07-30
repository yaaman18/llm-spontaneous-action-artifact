---
title: 
author: 
pages: 11
conversion_method: pymupdf4llm
converted_at: 2025-07-30T19:56:48.507250
---

# 

IEEE TRANSACTIONS ON INTELLIGENT VEHICLES 1

# Bi-AM-RRT*: A Fast and Efficient Sampling-Based Motion Planning Algorithm in Dynamic Environments


Ying Zhang, Heyong Wang, Maoliang Yin, Jiankun Wang, _Senior Member, IEEE,_
and Changchun Hua, _Senior Member, IEEE_



_**Abstract**_ **—The efficiency of sampling-based motion planning**
**brings wide application in autonomous mobile robots. The**
**conventional rapidly exploring random tree (RRT) algorithm**
**and its variants have gained significant successes, but there are**
**still challenges for the optimal motion planning of mobile robots**
**in dynamic environments. In this paper, based on Bidirectional**
**RRT and the use of an assisting metric (AM), we propose a**
**novel motion planning algorithm, namely Bi-AM-RRT*. Different**
**from the existing RRT-based methods, the AM is introduced**
**in this paper to optimize the performance of robot motion**
**planning in dynamic environments with obstacles. On this basis,**
**the bidirectional search sampling strategy is employed to reduce**
**the search time. Further, we present a new rewiring method**
**to shorten path lengths. The effectiveness and efficiency of**
**the proposed Bi-AM-RRT* are proved through comparative**
**experiments in different environments. Experimental results show**
**that the Bi-AM-RRT* algorithm can achieve better performance**
**in terms of path length and search time, and always finds near-**
**optimal paths with the shortest search time when the diffusion**
**metric is used as the AM.**


_**Index Terms**_ **—Mobile robot, motion planning, bidirectional**
**search, rewiring**


I. I NTRODUCTION


Recent advances in robotics have prompted an increasing
number of autonomous mobile robots to be used in various

fields, such as transportation [1], manufacturing [2], rescue

[3], domestic service [4], and so on. As a fundamental task
of mobile robots, motion planning aims to plan a feasible
collision-free path from the starting point to the goal point
for the robot in the working environment with static or
dynamic obstacles [5]. In such context, lots of research efforts
have been conducted on the motion planning problem. For
instance, based on the grid map, the Dijkstra [6] algorithm


This work was supported in part by the National Natural Science Foundation of China under Grant No. 62203378, 62203377, U22A2050, in part
by the Hebei Natural Science Foundation under Grant No. F2022203098,
F2021203054, in part by the Science and Technology Research Plan for
Colleges and Universities of Hebei Province under Grant No. QN2022077,
and in part by the Hebei Innovation Capability Improvement Plan Project
under Grant No. 22567619H. _(Corresponding author: Ying Zhang.)_
Y. Zhang, H. Wang, M. Yin, and C. Hua are with the School of
Electrical Engineering and the Key Laboratory of Intelligent Rehabilitation and Neromodulation of Hebei Province, Yanshan University, Qinhuangdao, 066004, China. (e-mail: yzhang@ysu.edu.cn; wtk0405@163.com;
yin924431601@163.com; cch@ysu.edu.cn).
J. Wang is with the Shenzhen Key Laboratory of Robotics Perception
and Intelligence, Shenzhen 518055 China, and also with the Department of
Electronic and Electrical Engineering, Southern University of Science and
Technology, Shenzhen 518055, China (e-mail: wangjk@sustech.edu.cn).



can derive a feasible trajectory by traversing the entire map.
In order to save computing resources, A* [7] and anytime
repairing A* (ARA*) [8] use a heuristic search strategy to
quickly obtain optimal solution. However, these methods are
not suitable for high-dimensional environments or differential
constraints. Moreover, to address dynamic obstacles, the D*

[9] and the anytime D* [10] are investigated to search for
feasible solutions in dynamic environments. The methods
above are grid-based algorithms that require discretization of
the state space, which leads to an exponential growth in time
spent and memory requirements with the increase of the state
space dimension [11]. To reduce the time cost and memory
usage, diffusion map is employed [12]. It is a non-linear
dimensionality reduction technique, and seeks for a feasible
solution by transforming each state on the map into a diffusion
coordinate [13]. Nevertheless, this treatment tends to ignore
some details in the environment, leading to poor planning
performance or even getting into trap in complex dynamic
environments.

For fast and high-quality motion planning in complex dynamic environments, sampling-based methods have attracted
significant attention. Typically, the rapidly exploring random
tree (RRT) algorithm [14] has been widely used and achieved
great success due to its efficiency and low memory usage.
To this end, many of its variants have been presented. For
example, RRT-connect [15] shortens the search time by exploiting goal bias and using two trees to search simultaneously. RRT* [16] adds a rewiring process to shorten the path
length. Extended-RRT [17] re-searches for new collision-free
path from the root when there are obstacles in the planned
trajectory. But this practice is time-spending. Besides, the RTRRT* [18] retains information about the whole tree from the
robot’s current position, and uses existing branches around
obstacles to locally plan feasible paths. However, the growth of
the whole tree takes more time. In such case, an extended RRTbased planning method with the assisting metric (AM) [19] is
investigated to guide the growth of the tree to shorten the path
search time. Although the utilization of AM can accelerate
the RRT exploration process, the search time and path length
needs to be improved in dynamic environments.
In this paper, we propose a novel motion planning method
based on bidirectional RRT and AM, namely Bi-AM-RRT*,
to reduce the search time and path length in dynamic environments. The presented Bi-AM-RRT* exploits the trunk
information of the reverse tree with the forward tree to


IEEE TRANSACTIONS ON INTELLIGENT VEHICLES 2



efficiently generate a feasible path to the goal position. Based
on this, the AM is used to improve the performance of motion
planning in environments with obstacles. The AM can be any
metric, such as Euclidean metric, diffusion metric, or geodesic
metric. Besides, in order to optimize the search path, a new
rewiring strategy based on the root and goal is presented to
shorten the path length. The main contributions of this work
include:


_•_ an AM-based bidirectional search sampling framework
for robot motion planning in dynamic environments;

_•_ a novelly fast and efficient motion planning algorithm,
namely Bi-AM-RRT*, to improve the motion planning
performance;

_•_ a new rewiring strategy to accelerate the path optimization process to reduce the path length;

_•_ evaluation and discussion on comparative experiments in
different environments, which demonstrate the validity
and efficiency of Bi-AM-RRT*.

The remainder of this paper is structured as follows. Section
II presents the related work. In Section III, the problem
definition and AM-RRT* are introduced. Section IV elaborates
the proposed Bi-AM-RRT*. Section V and Section VI describe
the extensive experiments and discuss the results, respectively.
Section VII concludes this paper.


II. R ELATED WORK


Robot motion planning aims at planning a feasible path for
robots, and has received significant attention over the years,
especially in dynamic environments. Many algorithms have
been proposed to address the motion planning problem.
To plan a feasible path, the artificial potential field algorithm
was introduced for robot motion planning [20], which uses the
direction of the fastest potential field decline as the moving
direction of the robot. However, when in an environment with
obstacles, such solution is prone to fall into local optimisation.
In recent years, the learning-based motion planning strategies
have been investigated. Everett _et al_ . [21] proposed an obstacle avoidance method that trains in simulation with deep
reinforcement learning (RL) without requiring any knowledge
of other agents’ dynamics. Similarly, Wang _et al_ . [22] designed
an RL-based local planner, which adopts the global path as the
guide path to adapt to the dynamic environment. To optimize
the planner, Pérez-Higueras _et al_ . [23] combined inverse
reinforcement learning with RRT* to learn the cost function.
The introduction of a machine learning improves the agent’s
path planning and obstacle avoidance performance in dynamic
environments. Notably, these learning-based methods need to
train the model in advance, which is time-spending. Moreover,
in order to find the optimal trajectory, grid-based motion
planning research efforts have been conducted extensively.
For example, based on the grid map, A* [7] was used to
search for feasible solutions and gained great success. Koenig
_et al_ . presented D*-lite for robot planning in unknown terrain
based on the lifelong planning A*. The performance is closely
related to the degree of discretization of the state space.
Although these grid-based approaches can always search for
the optimal path (if one exists), they do not perform well as



the scale of the problem increases, such as time-consuming
and high memory consumption.
To improve planning performance, sampling-based methods
are considered as a promising solution. In particular, RRTbased algorithms are widely popular due to their ability to
efficiently search state spaces and have proven to be an effective way to plan a feasible path for robots [24]. For instance,
Kuwata _et al_ . [25] proposed CL-RRT for motion planning
in complex environments. This method uses the input space
of a closed-loop controller to select samples and combines
effective techniques to reduce the computational complexity of
constructing a state space. Based on the probabilist collision
risk function, Fulgenzi _et al_ . [26] introduced a Risk-RRT
method. In this solution, a Gaussian prediction algorithm is
used to actively predict the moving obstacles and the sampled
trajectories to avoid collisions. To achieve dynamic obstacle
avoidance, Naderi _et al_ . [18] designed a RT-RRT* algorithm
that interweaves path planning with tree growth and avoids
waiting for the tree to be fully constructed by moving the
tree root with the agent. Analogously, Armstrong _et al_ . [19]
put forward an AM-RRT* by using AM to accelerate the
path planning process of RT-RRT*. However, the planning
performance is not satisfactory, especially in terms of search
time. In order to reduce the search time, bidirectional search
strategies are widely employed. As an early proposed bidirectional tree algorithm, RRT-connect [15] uses a greedy heuristic
to guide the growth of two trees, thereby shortening the search
time. Subsequently, other variants such as Informed RRT*connect [27], B2U-RRT [28], Bi-Risk-RRT [29], etc. were
proposed. They all incorporate bidirectional search strategy
and demonstrate the effectiveness of it. Inspired by AM-RRT*
and RRT-connect, a novel AM-based bidirectional search
sampling framework for motion planning, i.e., Bi-AM-RRT*,
is proposed in this paper to further reduces the search time
and the path length.
Additionally, it is necessary to obtain an optimal path while
maintaining the speed of the planner to guarantee the quality
of planning. To address the problem of path optimization,
Karaman _et al_ . [30] proposed RRT* by using newly generated nodes to rewire adjacent vertices to ensure asymptotic
optimality. But the convergence speed is slow. In order to
accelerate the convergence speed, Yi _et al_ . [31] suggested a
sampling-based planning method with Markov Chain Monte
Carlo for asymptotically-optimal motion planning. Chen _et_
_al_ . [32] designed DT-RRT to abandon the rewire process
and add re-search parent based on the shortcut principle.
Although this approach can speed up the convergence process,
it tends to produce suboptimal paths. Analogously, Wang _et al_ .

[33] presented a P-RRT*-connect algorithm to accelerate the
convergence of RRT* using an artificial potential field method.
Besides, based on the path optimization and intelligent sampling techniques, Islam _et al_ . [34] proposed RRT*-Smart,
which aims to obtain an optimum or near optimum solution.
Gammell _et al_ . [35] investigated the optimal sampling-based
path planning with a focused sampling method and presented
Informed RRT* to improve the covergence of RRT*. However,
these methods face challenges for the efficiency of motion
planning in dynamic environments.


IEEE TRANSACTIONS ON INTELLIGENT VEHICLES 3


In this paper, based on bidirectional search sampling strategy, a novel motion planning method, namely Bi-AM-RRT*,
is proposed with a new rewiring scheme to reduce the path
length and the search time for agent to find the goal in dynamic
environments.


III. P RELIMINARIES



In this section, the problem definition of motion planning
and the programs on which the algorithm depends are introduced first, and then the sampling-based planning algorithm
with AM, referred to as AM-RRT*, is described.


_A. Motion Planning Problem Definition_

Let us define the state space as _X ∈_ R _[d]_ . _X_ _obs_ _∈_ _X_ denotes
obstacles in the state space, while _X_ _free_ = _X/X_ _obs_ is the
free space without obstacles. _x_ _agent_ _∈_ _X_ _free_ is defined as the
state of the mobile robot in the space, and the goal state is
represented as _x_ _goal_ _∈_ _X_ _free_ . In this paper, a search tree _T_
_∈_ _X_ _free_ is used to generate a feasible collision-free path (i.e.,
points on the path _x_ _i_ _∈_ _T_ ) from the start point _x_ _root_ to the
goal point _x_ _goal_ . During exploration, the Bi-AM-RRT* can
grow both forward tree and reverse tree, which are denoted
_T_ _f_ and _T_ _r_, respectively. In addition, there are user-defined
the maximum edge length _e_ _max_ and the maximum number
of nodes _n_ _max_ in the circular domain with radius _e_ _max_ to
control the growth state of _T_ . Let _t_ _exp_ be the tree growth time.
Meanwhile, the root rewiring time and goal rewiring time are
denoted as _t_ _root_ and _t_ _goal_, respectively. When the Euclidean
distance is less than _σ_ and there is no obstacle between forward

tree and reverse tree, two trees can be joined as one tree, where
_σ_ represents the connecting distance of two trees.
In the presented method, AM _d_ _A_ can be the Euclidean
metric, the diffusion metric [13], and the geodesic metric [36],
which are indicated as _d_ _E_, _d_ _D_, and _d_ _G_, respectively. These
metrics are used to calculate the distance information of two

states in the state space. Specifically, the Euclidean distance
is expressed as


_d_ _E_ ( _x_ _a_ _, x_ _b_ ) = _∥x_ _a_ _, x_ _b_ _∥._ (1)


The Euclidean distance between the two points is obtained
based on the L2 norm of _x_ _a_ and _x_ _b_ . The diffusion distance is
yielded by calculating the Euclidean distance of the approximate diffusion coordinates corresponding to each of the two
states, and is described as


_d_ _D_ ( _x_ _a_ _, x_ _b_ ) = _∥h_ ( _g_ ( _x_ _a_ )) _, h_ ( _g_ ( _x_ _b_ )) _∥_ (2)


where _g(·)_ refers to mapping a state _·_ in the grid to the nearest
point, and _h(·)_ is mapping a state _·_ to the diffuse coordinates.
_d_ _D_ can provide a good approximation when an obstacle is
present. The geodesic distance is the use of the Dijkstra [6]
method to generate a distance matrix from the connection
matrix of discretization state space. It has the advantage of
high precision, but is time-spending.
Next, the procedures on which the algorithm depends are
described [19]. _Cost(T, x)_ refers to the length of the path from
the root to _x_ in _T_ based on Euclidean distance. _Path(T, x)_ refers
to returning the sequence of path nodes from the root to _x_ in



(a) (b)


Fig. 1. The obstacle avoidance process of AM-RRT*. (a) The branch
information (green circular area) is used for motion planning, and (b) obstacle
avoidance when encountering a dynamic obstacle, where red represents the
agent, blue represents the goal point, and black represents the obstacle.


_T_ . _FreePath(x_ _a_ _, x_ _b_ _)_ returns true when there are no obstacles
between x _a_ and x _b_ . _Nearest(T, x)_ returns the nearest neighbor
to _x_ if there is no obstacle between _x_ and its nearest neighbor.
_RewireEllipse(T, x_ _goal_ _)_ is to return the state set within the
rewire ellipse [35]. _Enqueue(Q, x)_ is the addition of _x_ to
the end of _Q_ . _Dequeue(Q)_ refers to removing and returning
the first item in _Q_ . _Push(S, x)_ is to add _x_ to the front of _S_ .
_Nearby(T,x)_ returns the set of all nodes within E-distance e _max_
of _x_ . _Pop(S)_ refers to deleting and returning the first item in
_S_ . _Second(S)_ means that the second item in _S_ is returned but
not removed. _UpdateEdge(T, x_ _new_ _, x_ _child_ _)_ replaces the edge
_(x_ _parent_ _, x_ _child_ _)_ with _(x_ _new_ _, x_ _child_ _)_ in _T_, where _x_ _parent_ is the
parent of _x_ _child_ in _T_ . _Len(X)_ returns the queue length of _X_ .


_B. AM-RRT*_


AM-RRT* is an informed sampling-based planning algorithm with AM [19]. Typically, AM-RRT* uses the diffusion
distance as an AM, which is derived from the diffusion map

[12] and is also a kind of grid map. It utilizes a dimensional
collapse method to reduce time and memory consumption.
Although the diffusion distance alone performs poorly in
complex environments, but it can achieve good performance
as an AM of RRT*, and can quickly find collision-free paths
when obstacles appear. Fig. 1 shows the obstacle avoidance
performance with AM-RRT*. When the obstacle appears on
the path, AM-RRT* does not regenerate the tree, but uses the
information in the whole tree for obstacle avoidance action,
especially the node information around the obstacle. As can
be viewed in Fig. 1, when obstacles appear, with the help of
diffusion metric, a feasible path can be quickly drawn based
on node rewiring by using the branch information of the tree
below the original path. During agent movement, the tree is
maintained in real time. At the same time, the planned path is
also rewired for optimization, and the path lengths are made
approximately optimal by successive iterations. The whole
process of tree growth is similar to RRT* and its variants.
In this process, the diffusion map only plays a leading role in
guiding the tree to find the goal point quickly and cover the full
space faster while maintaining its probabilistic completeness
in a complex environment [19].


IEEE TRANSACTIONS ON INTELLIGENT VEHICLES 4


**Algorithm 1:** Bi-AM-RRT*


**Input:** _Agent_, _Goal f_, _Map_, _Q f_ _root_ _←_ [], _Q r_ _root_ _←_ [],
_Q f_ _goal_ _←_ [], _S f_ _goal_ _←_ []
**Output:** _Path_

**1** _Path←_ _φ_ ; _T_ _f_ _←_ _φ_ ; _T_ _r_ _←_ _φ_ ;

**2** _load()←Map_ ;

**3** **while** _Agent /∈_ _x_ _goal f_ **do**

**4** _load()←X_ _free_ _, X_ _obs_ ;

**5** _x_ _agent_ _←Agent_, _x_ _root f_ _←Agent_, _x_ _goal f_ _←Goal_ ;

**6** _x_ _root r_ _←Goal_, _x_ _goal r_ _←x_ _root f_ (time=0);


_̸_



_̸_



_̸_



(a) (b)


Fig. 2. Bidirectional tree growth rewiring process. (a) The forward tree (in _̸_
blue) and the reverse tree (in green) grow at the same time. (b) When the two
trees are close enough to connect into one tree, the reverse tree stops growing
and initializes.


IV. P ROPOSED B I -AM-RRT*


This paper proposes the Bi-AM-RRT* for real-time optimal
motion planning of mobile robots in dynamic environments.
Generally, our proposed Bi-AM-RRT* uses bidirectional trees
(i.e., forward and reverse trees) for searching and accelerates
the path optimization by a new rewiring process. In this
section, the details of the of our Bi-AM-RRT*, especially
the proposed bidirectional search strategy and the new path
rewiring strategy, are presented.


_A. Bidirectional Search Strategy for Bi-AM-RRT*_


For RRT-based motion planning, the use of a bidirectional
search strategy is faster to plan feasible paths than unidirectional. Fig. 2 illustrates the bidirectional tree growth rewiring
process. First, the two trees grow simultaneously. When the
forward tree and the reverse tree are meet, the forward tree
uses the reverse tree to generate the path to the goal while the
reverse tree stops growing and initializes. Finally, the forward
tree continues to grow to the full map. In this process, the
branch information of the forward tree is used for obstacle

avoidance and path optimization (refer to Fig. 1).
Algorithm 1 describes the detail of Bi-AM-RRT*. First,
the forward tree and reverse tree information are initialized,

_∼_
and the map information is loaded (Lines 1 2). Then, the
goal points are set and the _X_ _free_ and _X_ _obs_ information is
continuously updated. The root state of the forward tree
follows the position state of the agent, and the goal state is
provided by someone. The root state of the reverse tree is
set to the goal state, while the goal state is set to the initial
state of the agent position and does not change as the agent

_∼_
position moves (Lines 3 6). When two trees are not connected
(i.e., _x_ _goal f_ _∈_ _T_ _r_ _, x_ _goal f_ _/∈_ _T_ _f_ ), they grow simultaneously.
When two trees are connected successfully, only the forward
tree continues to expand to the full map, generating more
nodes to optimize the path to avoid obstacles or make the
path shorter (Lines 8 _∼_ 11). The function _Meet_ ( _T_ _f_, _T_ _r_ ) denotes
that true is returned when the Euclidean distance between _T_ _f_
and _T_ _r_ is less than the connection distance _σ_ and there is
no obstacle blocking it. If true is returned, the information of
the reverse tree is fused to the forward tree by the function
_Swap_ ( _T_ _f_, _T_ _r_ ). Subsequently, the reverse tree stops expanding



**7** _start←clock()_ ;

**8** **while** _clock() - start<t_ _exp_ _∨_ _x_ _goal f_ = _̸_ _φ_ **do**

**9** _T_ _f_ _←Expend f(T_ _f_ _, Q f_ _root_ _, Q f_ _goal_ _, S f_ _goal_ _,_
_x_ _goal f_ _)_ ;

**10** **if** _x_ _goal f_ _/∈_ _T_ _f_ **then**

**11** _T_ _r_ _←Expend r(T_ _r_ _, Q r_ _root_ _, x_ _root r_ _)_ ;


**12** **if** _Meet(T_ _f_ _,T_ _r_ _)_ **then**

**13** _Swap(T_ _f_ _,T_ _r_ _)_ ;

**14** _T_ _r_ _←init()_ ;


**15** _Path=Path f(T_ _f_ _, Nearest(T_ _f_ _, x_ _goal f_ _))_ ;

**16** Move Agent towards _x_ _goal f_ ;


_∼_
and initializes (Lines 12 14). And a collision-free path to the
goal is generated, and finally, the agent moves along that path

_∼_
(Lines 15 16). When the agent reaches the goal, it waits for
information about the next goal. The above steps are repeated
once a new goal is given.
The tree is grown in a way that maintains the probabilistic completeness of random sampling while using AM for
guidance, which make the tree growth more aggressive and
efficient. The growth of the forward tree is presented in Algorithm 2. When the goal point is given (Line 1), the forward
tree actively grows toward the goal point under the guidance
of the AM. The entire space is then covered by continuous
sampling process using the function _SampleState_ (T _f_, x _goal f_ )
(Line 2). _SampleState_ (T _f_, x _goal f_ ) returns the sampling set _X_ _s_,
which is defined as



_̸_


_X_ _s_ =



_̸_









_̸_


_{x_ _goal_ _}_ _p >_ 0 _._ 7 and _x_ _goal_ _/∈_ _T_
_X_ _random_ _∈_ _X_ _free_ _p <_ 0 _._ 5 or _x_ _goal_ _/∈_ _T_
_RewireEllipse_ ( _T, x_ _goal_ ) otherwise



_̸_


where _p ∈_ [0,1). Afterwards, the root rewiring (see Algorithm
3) is performed to optimize the path (Line 3). In fact, the
root rewiring process is always performed. When the goal is
found, the rewiring of the goal point (see Algorithm 5) is
then implemented to further optimize the path (Lines 4-5). In
particular, the growth of the reverse tree is basically the same
as that of the forward tree. Since the reverse tree does not

need to reach its own goal, there is no goal rewiring step.


_B. Path Optimization With Rewiring Strategy_


To optimize the path, a new rewiring method based on RRT*
is proposed, which re-searches the grandfather node instead of
the parent node to speed up the convergence rate. Algorithm 3


IEEE TRANSACTIONS ON INTELLIGENT VEHICLES 5


**Algorithm 2:** Expend f

**Input:** _Goal_, _Map_, _Q f_ _root_, _Q f_ _goal_, _S f_ _goal_
**Output:** _T_ _f_

**1** _x_ _goal f_ _←Goal_ ;

**2** _T_ _f_ _←SampleState(T_ _f_ _, x_ _goal f_ _)_ ;

**3** _T_ _f_ _←RewireRoot(T_ _f_ _, Q f_ _root_ _)_ ;

**4** **if** _x_ _goal f_ _∈_ _T_ _f_ **then**

**5** _T_ _f_ _←RewireGoal(T_ _f_ _, Q f_ _root_ _, S f_ _goal_ _, x_ _goal f_ _)_ ;



provides the root rewiring process of the forward tree. _Q f_ _root_
is a reference queue used to find less costly points to update
_T_ _f_ . And the new _x_ _root f_ is the first data of _Q f_ _root_ . When the
root queue is empty, the information of the offset root is added
and the root queue is reset (Lines 1 _∼_ 2). Then, _t_ _root_ is used to

_∼_
limit the time of root rewiring (Lines 3 4). When the number
of data in _Q f_ _root_ is greater than 0 and less than or equal to 2,

_∼_
the root rewiring of AM-RRT* [19] is used (Lines 5 6). When
the number of data in _Q f_ _root_ is greater than 2, the proposed

_∼_
new rewiring method is executed for optimization (Lines 7 8).
In such optimization process, the path length is reduced by
re-searching for a point near the grandfather node that is less
costly and has no obstacle between it and the child node as the
new parent node, as shown Fig. 3, which is given in Algorithm
4. The combination of these two rewiring methods accelerates
the convergence speed of path optimization and avoids the
generation of suboptimal paths at the corners. Thus, the path
length is shortened. The root rewiring process in the reverse
tree expansion process is the same as that in the forward tree.


**Algorithm 3:** RewireRoot


**Input:** _Q f_ _root_
**Output:** _T_ _f_

**1** **if** _Q f_ _root_ _= φ_ **then**

**2** _Enqueue(Q f_ _root_ _, x_ _root f_ _)_ ;

**3** _start←clock()_ ;

**4** **while** _clock() - start<t_ _root_ **do**

**5** **if** _0<len(Q f_ _root_ _)<=2_ **then**

**6** _RewireRootFirst(T_ _f_ _, Q f_ _root_ _)_ ;


**7** **if** _len(Q f_ _root_ _)>2_ **then**

**8** _RewireRootSecond(T_ _f_ _, Q f_ _root_ _)_ ;


Algorithm 4 summarizes the optimization process when
_len(Q f_ _root_ _)>2_ . In this case, rewiring uses not just the information of _x_ _r_ 1 but both _x_ _r_ 1 and _x_ _r_ 2 to speeds up the path
optimization process and reduces the path length. _x_ _r_ 1 and _x_ _r_ 2
are dequeued in sequence and try to find the point in _x_ _r_ 1
nearest neighbor that can reduce the path cost. If it exists, the
_T_ _f_ is updated (Lines 1 _∼_ 8). If _x_ _near_ is not in reference queue
_Q f_ _root_, it is added to the _Q f_ _root_ (Lines 9 _∼_ 10).
When the goal point is in the tree, the goal rewiring method
is performed, which is presented in Algorithm 5. In the
algorithm, the reference data stack _S f_ _goal_ and reference queue
_Q f_ _goal_ are used for path optimization, where _S f_ _goal_ stores the
nodes of the current branch and _Q f_ _goal_ stores the nodes of



Fig. 3. The path optimization process. The tree path is further optimized to
A-B on the right when there is a less costly proximity point b around point
C. Although the C-D path is better when there is a less costly proximity point
a around point E, the path is not optimized due to the obstacle (black square)
blocking it.


**Algorithm 4:** RewireRootSecond


**Input:** _Q f_ _root_
**Output:** _T_ _f_

**1** _x_ _r_ 1 _←Dequeue(Q f_ _root_ _)_ ;

**2** _x_ _r_ 2 _←Dequeue(Q f_ _root_ _)_ ;

**3** **for** _x_ _near_ _∈_ _Nearby(T_ _f_ _, x_ _r_ 2 _)_ **do**

**4** **if** _FreePath(x_ _near_ _, x_ _r_ 1 _)_ **then**

**5** _c_ _old_ _←Cost(T_ _f_ _, x_ _r_ 1 _)_ ;

**6** _c_ _new_ _←Cost(T_ _f_ _, x_ _r_ 1 _)+d_ _E_ _(x_ _r_ 2 _, x_ _near_ _)+d_ _E_ _(x_ _r_ 1 _,_
_x_ _r_ 2 _)_ ;

**7** **if** _c_ _new_ _<c_ _old_ **then**

**8** _T_ _f_ _←UpdateEdge(T_ _f_ _, x_ _near_ _, x_ _r_ 1 _)_ ;


**9** **if** _x_ _near_ _/∈_ _Q f_ _root_ **then**

**10** _Enqueue(Q f_ _root_ _, x_ _near_ _)_ ;


the next branch. When both _S f_ _goal_ and _Q f_ _goal_ are empty,
the root information of the real-time tree offset is pushed to

_∼_
_S f_ _goal_ (Lines 1 2). Here a two-step optimization approach is
introduced in this paper: (1) when time is less than _t_ _goal_ and
there is a non-empty set of _len(Q f_ _goal_ _)_ or _len(S f_ _goal_ _)_, the

_∼_
goal rewiring of AM-RRT* [19] is performed (Lines 4 6);
and (2) when the time is less than twice _t_ _goal_ and as long as
there is a set _len(Q f_ _goal_ _)_ or _len(S f_ _goal_ _)_ longer than 2, the
optimization strategy is performed according to Fig. 3 (Lines
7 _∼_ 9), and the details can be found in Algorithm 6. The twostep optimization approach avoids suboptimal paths caused by
obstacles, which reduce path length.
Algorithm 6 elaborates the second step of the optimization
in Algorithm 5. This process uses the information from both
points _x_ _r_ 1 and _x_ _r_ 2 to speed up the optimization process and
reduce the path length. When the length of _S f_ _goal_ is greater
than 2, _x_ _r_ 1 and _x_ _r_ 2 are popped in turn, otherwise they exit the
queue in turn. (Lines 1 _∼_ 6). When _x_ _r_ 1 is inside the rewire
ellipse [35] (nodes inside the ellipse are more likely to be
utilized), the cost of each node within the _x_ _r_ 1 radius of _e_ _max_
is calculated. And if there is a point with a smaller cost,

_∼_
the rewiring optimization is performed (Lines 7 13). If the
point is not in _S f_ _goal_, it will be added to _S f_ _goal_ and _Q f_ _goal_ .
Moreover, if the distance from the second node at the top in
_S f_ _goal_ to the goal point is greater than the sum of the distance
from _x_ _r_ 1 to the goal point and the distance from _x_ _r_ 1 to _x_ _r_ 1,
the branch is discarded. Then the next iteration is continued


IEEE TRANSACTIONS ON INTELLIGENT VEHICLES 6



**Algorithm 5:** RewireGoal


**Input:** _Q f_ _goal_ _, S f_ _goal_ _, x_ _goal f_
**Output:** _T_ _f_

**1** **if** _Q f_ _goal_ _∨_ _S f_ _goal_ _= φ_ **then**

**2** _Push(S f_ _goal_ _, x_ _root f_ _)_ ;

**3** _start←clock()_ ;

**4** **while** _clock() - start<t_ _goal_ **do**

**5** **if** _len(Q f_ _goal_ _)>0 ∧_ _len(S f_ _goal_ _)>0_ **then**

**6** _RewireGoalFirst(T_ _f_ _, Q f_ _goal_ _, S f_ _goal_ _,_
_x_ _goal f_ _)_ ;


**7** **while** _t_ _goal_ _<clock() - start<2·t_ _goal_ **do**

**8** **if** _len(Q f_ _goal_ _)>2 or len(S f_ _goal_ _)>2_ **then**

**9** _RewireGoalSecond(T_ _f_ _, Q f_ _goal_ _, S f_ _goal_ _,_
_x_ _goal f_ _)_ ;


_∼_
(Lines 14 18).


**Algorithm 6:** RewireGoalSecond


**Input:** _Q f_ _goal_ _, S f_ _goal_ _, x_ _goal f_
**Output:** _T_ _f_

**1** **if** _len(S f_ _goal_ _)>2_ **then**

**2** _x_ _r_ 1 _=Pop(S f_ _goal_ _)_ ;

**3** _x_ _r_ 2 _=Pop(S f_ _goal_ _)_ ;


**4** **else**


**5** _x_ _r_ 1 _=Dequeue(Q f_ _goal_ _)_ ;

**6** _x_ _r_ 2 _=Dequeue(Q f_ _goal_ _)_ ;


**7** **if** _x_ _r_ 1 _∈RewireEllipse(x_ _goal f_ _)_ **then**

**8** **for** _x_ _near_ _∈_ _Nearby(T_ _f_ _, x_ _r_ 1 _)_ **do**

**9** **if** _FreePath(x_ _near_ _, x_ _r_ 2 _)_ **then**

**10** _c_ _old_ _←Cost(T_ _f_ _, x_ _near_ _)_ ;

**11** _c_ _new_ _←Cost(T_ _f_ _, x_ _r_ 2 _)+d_ _E_ _(x_ _r_ 1 _, x_ _near_ _)_
_+d_ _E_ _(x_ _r_ 1 _, x_ _r_ 2 _)_ ;

**12** **if** _c_ _new_ _<c_ _old_ **then**

**13** _T_ _f_ _←UpdateEdge(T_ _f_ _, x_ _r_ 2 _, x_ _near_ _)_ ;


**14** **if** _x_ _near_ _/∈_ _S f_ _goal_ **then**

**15** _S f_ _goal_ _←x_ _near_ _}_ ;

**16** _Q f_ _goal_ _←x_ _near_ _}_ ;


**17** **if** _len(S f_ _goal_ _)>1 ∨_ _d_ _A_ _(Second(S f_ _goal_ _), x_ _goal f_ _)_
_>d_ _A_ _(x_ _r_ 1 _, x_ _goal f_ _)+d_ _A_ _(x_ _r_ 1 _, x_ _r_ 2 _)_ **then**

**18** _S f_ _goal_ _=[]_


The proposed Bi-AM-RRT* can significantly reduce planning costs in both small simple environments and large
complex environments with dynamic obstacles. The use of
bidirectional tree shorten the time cost of finding the feasible
path. During exploration, the suboptimal paths resulting from
bidirectional tree connection are optimized by growing the
entire path radially around. In addition, the use of the proposed
root rewiring and goal rewiring methods accelerates path
optimization and reduces the path length.



V. E XPERIMENTS AND R ESULTS

In order to prove the effectiveness and efficiency of the proposed method, extensive comparative experiments are carried
out in different simulation environments. This section gives
the experimental details, while the comparison and discussion
of experimental results are provided.


_A. Experimental setting_

The experiments are conducted in PyCharm 2021 on top of
a Lenovo Y7000p laptop running Windows OS Intel i5-8300H
CPU at 2.3 GHz having 16 GB of RAM. To demonstrate the
validity and efficiency of the proposed method, our method
is compared with RT-RRT* [18] and AM-RRT* [19]. Further,
based on the bidirectional search sampling strategy and new
rewiring strategy proposed in this work, extensive comparative
experiments are designed using five state-of-the-art planners
RT-RRT*, RT-RRT*(D), AM-RRT*(E), AM-RRT*(D) and
AM-RRT*(G) [18], [19] to fully evaluate the performance of
the Bi-AM-RRT*. Specifically,

1) based on five planners, only the bidirectional search
strategy is used to design five types of planners, which are
denoted as RT-RRT*-1, RT-RRT*(D)-1, AM-RRT*(E)-1,
AM-RRT*(D)-1 and AM-RRT*(G)-1.
2) based on five planners, only the proposed rewiring strategy is used to design five types of planners, which are
denoted as RT-RRT*-2, RT-RRT*(D)-2, AM-RRT*(E)-2,
AM-RRT*(D)-2 and AM-RRT*(G)-2.
3) based on five planners, both the bidirectional search
strategy and proposed rewiring strategy are used to design
five types of planners, which are denoted as Bi-RT-RRT*,
Bi-RT-RRT*(D), Bi-AM-RRT*(E), Bi-AM-RRT*(D) and
Bi-AM-RRT*(G).

As shown in Table I, a total of 20 planners are implemented
for comparison. Moreover, experiments are carried out in three
challenging scenarios to better demonstrate the robustness and
applicability, namely Bug trap, Maze, and Office (see Fig. 4),
where the size of Bug trap and Maze is 100 _m ×_ 100 _m_, and
the size of Office is 200 _m ×_ 200 _m_ . In the three scenarios, the
parameter settings of planners are listed in Table II. Note that
the connection distance _σ_ used in bidirectional tree is set to

50 _m_ in the Bug trap scenario and 30 _m_ in the other scenarios.
In the experiment, each planner is tested in a typical task
where the agent needs to plan a feasible path to the goal point
G from the starting point S in different scenarios with static
obstacles, while recording the search time cost and path length
of the agent’s movement path from the start to the goal. To
fairly evaluate the performance of the method, each experiment
is repeated 25 times. The average of the 25 experiments is then
used for an unbiased comparison of experimental results. In
addition, we further verify the performance of the proposed
method in the environment with dynamic obstacles, where
dynamic obstacles are simulated by using black circles to
block the robot’s direction of motion (refer to Fig. 1).


_B. Results_

_1) Scenario With Static Obstacles:_ According to the experimental setup, 20 different planners were implemented in three


IEEE TRANSACTIONS ON INTELLIGENT VEHICLES 7


TABLE I

D ESCRIPTION OF DIFFERENT PLANNERS


Scheme Planner


Original RT-RRT* RT-RRT*(D) AM-RRT*(E) AM-RRT*(D) AM-RRT*(G)

Bidirectional search-based RT-RRT*-1 RT-RRT*(D)-1 AM-RRT*(E)-1 AM-RRT*(D)-1 AM-RRT*(G)-1

Proposed rewiring strategy-based RT-RRT*-2 RT-RRT*(D)-2 AM-RRT*(E)-2 AM-RRT*(D)-2 AM-RRT*(G)-2

Bidirectional-and proposed rewiring strategy-based Bi-RT-RRT* Bi-RT-RRT*(D) Bi-AM-RRT*(E) Bi-AM-RRT*(D) Bi-AM-RRT*(G)


(a) (b) (c)


Fig. 4. Experimental scenario: (a) Bug trap, (b) Maze, and (c) Office, where the letters S and G represent the starting point and goal point, respectively. The
sizes of the three scenarios are 100 _m ×_ 100 _m_, 100 _m ×_ 100 _m_, and 200 _m ×_ 200 _m_, respectively.



TABLE II

P ARAMETERS SETTING OF PLANNER


_t_ _exp_ /s _t_ _root_ /s _t_ _goal_ /s _e_ _max_ /m _n_ _max_ _σ_ /m


RT-RRT* 0.15 0.003 0.003 5 12 30/50

RT-RRT*(D) 0.15 0.003 0.003 5 12 30/50

AM-RRT*(E) 0.15 0.002 0.004 5 20 30/50

AM-RRT*(D) 0.15 0.002 0.004 5 20 30/50

AM-RRT*(G) 0.15 0.002 0.004 5 20 30/50


different scenarios. The experimental results are shown in Fig.
5. Fig. 5(a) presents the performance comparison comparison
with and without the bidirectional search sampling strategy.
Since suboptimal paths can be generated in bidirectional tree
connections (see Fig. 6), the path length of the five planners
based on bidirectional strategy increases, but only by 0.8%.
Note that the search times are significantly improved with
the use of the bidirectional search strategy. In the Bug trap,
Maze, and Office, the time costs are reduced by about 69%,
40.1%, and 41.7%, respectively. In particular, the search time
of AM-RRT*(E)-1 can be reduced by up to 75.6% in the
Bug trap scenario. Therefore, the results illustrate that the use
of bidirectional search sampling strategy is effective.

The results of Fig. 5(b) demonstrate that the combination
of the original method and the proposed rewiring strategy can
optimize the average performance of the path length and search
time in the three scenarios to a certain extent. And the path
length can be reduced by an average of about 2.2%. Especially
for AM-RRT*(D)-2, it can still shorten the path length by 3%
and reduce the search time by 5.6% even in the large scenario
(i.e., Office). Overall, the results demonstrate the effectiveness
and generalizability of the proposed rewiring method of this

paper.

Fig. 5(c) illustrates a comparison of the results between the



TABLE III

M AP PROCESSING TIME FOR EACH PLANNER IN DIFFERENT SCENARIOS


Bug trap Maze Office


Bi-RT-RRT* / / /

Bi-RT-RRT*(D) 1.5s 1.5s 5.6s

Bi-AM-RRT*(E) / / /

Bi-AM-RRT*(D) 1.5s 1.5s 5.6s

Bi-AM-RRT*(G) 49s 49s 232s


solution presented in this paper (i.e., the strategy of fusing
bidirectional search sampling strategy and proposed rewiring
strategy) and the original solution. It can be seen that the
proposed solution can achieve superior performance in terms
of path length and search time, except for the slight increase
in path length of Bi-RT-RRT* and Bi-AM-RRT*(E) planners
in the Bug trap scenario. The reason for this is that the
bidirectional strategy and greater connection distance reduce
the search time, but when the goal point is found, the number
of nodes generated in the tree is insufficient, resulting in a
lower degree of path optimization, which will be discussed
in detail in Section VI. Besides, on average, Bi-AM-RRT*(E)
achieves the superior optimization performance in terms of
search time, which can be reduced by 76.7%, but increased
by 2.6%in terms of path length. It is worth noting that Bi-AMRRT*(D) obtains the most promising performance overall. In
the three scenarios, Bi-AM-RRT*(D) optimizes search time
by 24.6%, 45.2% and 44.9%, respectively, and reduces path
length by 1.4%, 0.7% and 2.8%, respectively.

In addition, for Bi-RT-RRT* and Bi-AM-RRT*(E) planners,
map processing time is not required. But for Bi-RT-RRT*(D)
and Bi-AM-RRT*(D) planners, the diffusion maps are needed,
while the geodesic metric is required for Bi-AM-RRT*(G)
planner. Table III shows the map processing time for each


IEEE TRANSACTIONS ON INTELLIGENT VEHICLES 8



Bug_trap



Maze



Office



210





135



305



RT-RRT* RT-RRT*(D) AM-RRT*(E) AM-RRT*(D) AM-RRT*(G) RT-RRT* RT-RRT*(D) AM-RRT*(E) AM-RRT*(D) AM-RRT*(G)



132.73 128.25 130.19 124.75 122.73 293.61 299.87 290.06 286.33 282.56









20

















1.6



50



195


25


200


20



Bi-RT-RRT* Bi-RT-RRT*(D) Bi-AM-RRT*(E) **Bi-AM-RRT*(D)** Bi-AM-RRT*(G) ~~Modify~~ Bi-RT-RRT* Bi-RT-RRT*(D) Bi-AM-RRT*(E) **Bi-AM-RRT*(D)** Bi-AM-RRT*(G) ~~Modify~~ Bi-RT-RRT* Bi-RT-RRT*(D) Bi-AM-RRT*(E) **Bi-AM-RRT*(D)** Bi-AM-RRT*(G)





3.23 0.94 4.28 0.43 0.38 0.8 0.57 0.82 0.17 0.16 16.63 14.68 24.63 0.49 0.47









2.94 0.96 4.49 0.39 0.36 0.79 0.54 0.84 0.17 0.17 17.36 15.02 24.97 0.5 0.48



Office



Bug_trap



(a)


Maze



135



300



















182.5 181.86 176.86 175.68 173.47 128.2 126.73 127.46 123.11 121.46 285.02 292.06 280.81 277.11 276.35



1.6



50





















8.02 1.7 19.34 0.53 0.45 1.15 0.98 1.45 0.29 0.21 30.14 24.6 41.14 0.84 0.79



Office



Bug_trap



(b)


Maze



135



300



RT-RRT*193.02 RT-RRT*(D)186.59 AM-RRT*(E)178.32 AM-RRT*(D)178.68 AM-RRT*(G)175.19 ~~Original~~ RT-RRT*131.99 RT-RRT*(D)127.42 AM-RRT*(E)129.79 AM-RRT*(D)124.01 AM-RRT*(G)122.43 ~~Original~~ RT-RRT*292.54 RT-RRT*(D)297.83 AM-RRT*(E)288.01 AM-RRT*(D)285.71 AM-RRT*(G)281.86

Bi-RT-RRT* Bi-RT-RRT*(D) Bi-AM-RRT*(E) **Bi-AM-RRT*(D)** Bi-AM-RRT*(G) ~~Modify~~ Bi-RT-RRT* Bi-RT-RRT*(D) Bi-AM-RRT*(E) **Bi-AM-RRT*(D)** Bi-AM-RRT*(G) ~~Modify~~ Bi-RT-RRT* Bi-RT-RRT*(D) Bi-AM-RRT*(E) **Bi-AM-RRT*(D)** Bi-AM-RRT*(G)













195.75 182.69 183.02 176.24 173.9 128.77 127.35 128.78 123.2 121.54 285.52 293.06 283.81 277.71 277.35



1.6



50



RT-RRT* RT-RRT*(D) AM-RRT*(E) AM-RRT*(D) AM-RRT*(G) RT-RRT* RT-RRT*(D) AM-RRT*(E) AM-RRT*(D) AM-RRT*(G) RT-RRT* RT-RRT*(D) AM-RRT*(E) AM-RRT*(D) AM-RRT*(G)













(c)


Fig. 5. Comparison of experimental results. The average path length and search time required by different planners to find a feasible path from the starting
point S to the goal point G in different scenarios, where (a) represents the results with (i.e., Modify) and without (i.e., Original) the bidirectional search
sampling strategy, and (b) represents the results with (i.e., Modify) and without (i.e., Original) the proposed rewiring strategy, and (c) represents the results
with (i.e., Modify) and without (i.e., Original) the bidirectional search sampling strategy and proposed rewiring strategy.


IEEE TRANSACTIONS ON INTELLIGENT VEHICLES 9


VI. D ISSCUSSION



(a) (b)


Fig. 6. Path optimization process. (a) When a bidirectional tree connection
produces a suboptimal path (red zone), (b) it can be optimized due to the
continuous growth of the tree.


planner in different scenarios. Although diffusion map processing takes time, the map processing time for Bug trap
and Maze scenarios is about 1.5 _s_ . Even for larger Office
scenario, it only takes 5.6 _s_ . As tested in this work, the geodesic
metric for Bug trap and Maze scenarios take about 49 _s_, while
Office scenarios take about 232 _s_ . In this context, Bi-AMRRT*(D) outperforms other planners in terms of total search
time (including map processing time) and path length. The
reason behind this is that diffusion maps are a way to use
dimensional collapse to reduce map processing time [12], [13],
such that some details are ignored when processing larger and
more complex maps. Therefore, in the Office scenario, the
search time of Bi-RT-RRT*(D) is still large, but it has less
impact on planners based on AM. Although the comprehensive
performance of Bi-RT-RRT*(D) in small scenes is similar to
that of Bi-AM-RRT*(D), it is not suitable for larger scenarios.
In conclusion, Bi-AM-RRT*(D) is an excellent planner that
further improves performance, and is suitable for both small
and large scenarios. The results, then, further demonstrate the
effectiveness and efficiency of our proposed strategy.


_2) Scenario With Dynamic Obstacles:_ In order to test
the obstacle avoidance performance of the proposed method,
the experiment is conducted in the Office scenario with the
dynamic obstacle. The dynamic obstacle is simulated by using
a solid black circle that can be added anywhere at any time
to block the path of the robot. The experimental results are
depicted in Fig. 7. When the goal point is given, both trees
grow at the same time [Fig. 7(a)]. When the distance is close
enough [Fig. 7(b)], the two paths are connected to one path
at two green dots, and the reverse tree stops growing and
initializes. The forward tree uses information from the reverse

path to grow quickly to the goal point and to the whole map.
During the navigation, when there is an obstacle in the path

[refer to Fig. 7(c)], the forward tree uses the node information
near the obstacle to quickly generate a feasible path to avoid
the obstacle, allowing the agent to move along the planned
path and safely reach the goal [see Fig. 7(d)]. Hence the
results show that the proposed method can address the obstacle
avoidance in dynamic environments.



This section discusses the effect of the connection distance

_σ_ on the obstacle avoidance performance and the number of
nodes on the path optimization.
In this paper, the values of _σ_ are set to 30 _m_ and 50 _m_ .
This is very large for the connection distance between the two
trees. While this makes it easier to connect the two trees, it
also makes it easier to produce the suboptimal path, as shown
in Fig. 6. For example, in the Bug trap scenario, _σ_ is set to
50 _m_ to allow the two trees of the Bi-RT-RRT* and Bi-AM
RRT*(E) planners to connect more easily, as illustrated in Fig.
8. Due to the guidance of Euclidean distance, the two trees are
trapped at point A and point B in Fig. 8(a) for a long time,
and the distance is far away. This is why the search times
of these two planners are optimized by more than 50%, as
shown in Fig. 5(a) and (c). When two trees are connected by
the diffusion map or geodesic metric, the other three planners
are connected basically in the upper left area of the Bug trap
scenario (see Fig. 2). Although _σ_ is set at 50 _m_, it is not fully
utilized, resulting in less optimization of the search time.
Also taking Bug trap scenario as an example, it can be seen
from Fig. 5 that the search time using AM-RRT*(E) is the
longest, but the path length is shorter than that of RT-RRT* and
almost the same as that of AM-RRT*(D). Since the agent starts
moving when the planner finds the goal point, AM-RRT*(E)
can generate more nodes in the more search time. In other
words, the path can be fully optimized by the time the agent
starts moving. Although the tree grows in real time, RT-RRT*
does not have enough nodes to path optimization, as depicted
in Fig. 9. In the experiment, when a feasible path to the goal
point is found, RT-RRT* generates an average of 445 nodes,
AM-RRT*(E) generates an average of 1062 nodes, and AMRRT*(D) generates only 76 nodes on average. After fusing
the bidirectional strategy, Bi-RT-RRT* and Bi-AM-RRT*(E)
can improve the efficiency of search time by more than 60%,
but lead to a slight increase in path length. Although Bi-AMRRT*(G) achieves the shortest search time and path length,
it requires a long map processing time. Bi-AM-RRT*(D), on
the other hand, completes the near-optimal path planning in a
less time.

In the Bug trap scenario, _σ_ is set to 50 _m_, and all planners can achieve obstacle avoidance performance. Although
the three planners do not take full advantage of this large
connection range, the Bi-RT-RRT* and Bi-AM-RRT*(E) can
maintain path optimization and obstacle avoidance functions.
This is due to the loopback path generated after the paths are
connected, and there are more nodes available in this region to
grow the tree and optimize the structure of the tree by rewiring
before the agent arrives. In other scenarios, however, setting _σ_
to 50 _m_ does not maintain rewiring and obstacle avoidance in
some extreme connection situations, as shown in Fig. 10. To
this end, the _σ_ setting of 30 is tested in the Office scenario,
which shows that excellent performance can be maintained
even under extreme connection conditions. For this purpose,
the _σ_ is set to 30 _m_ in experiments. The value of _σ_ should
be determined based on factors such as the size of the scene

map, the tree growth time _t_ _exp_, and the movement speed of


IEEE TRANSACTIONS ON INTELLIGENT VEHICLES 10


(a) (b) (c) (d)


Fig. 7. Obstacle avoidance performance of the proposed algorithm in the Office scenario. In (a) the blue line is the forward tree path and the green line is
the reverse tree path. When two trees are close enough, they are connected into one tree through two green points (b). And when the black circle of obstacle
appears in the path (c), a feasible path is quickly planned by using the information of nearby branches (d).



(a) (b)


Fig. 8. Path planned by Bi-RT-RRT* and Bi-AM-RRT*(E) in the Bug trap
scenario. Since the Euclidean metric guides, the forward tree will be trapped
in point A, and the reverse tree will be trapped in point B in (a) for a long
time, so _σ_ is set to 50 for optimization. Although there is a longer suboptimal
path after successful connection, but it has been gradually optimized before
the agent arrives (b).


(a) (b)


Fig. 9. Tree optimization process of the AM-RRT*(E) (a) and RT-RRT* (b).


the agent. In this paper, the values of _σ_ are not universal in
different scenarios, but have certain reference value.


VII. C ONCLUSION


In this paper, a novel motion planning approach, namely BiAM-RRT*, has been proposed. Bi-AM-RRT* uses a bidirectional search strategy and a new rewiring approach to reduce
the search time and the path length. In the Bi-AM-RRT*, two
trees grow simultaneously when the goal point is not in the
forward tree. Then they are connected as one tree when the



(a) (b)


Fig. 10. The obstacle avoidance of Bi-AM-RRT*(D) in the Office scenario
when _σ_ is set to 50 _m_ . Although the two trees are successfully connected,
there are no nodes between the two points A and B for tree growth (a). And
the growth rate of the two points A and B, is not enough to maintain the path
optimization and obstacle avoidance in that connected path when the agent
arrives, resulting in the agent colliding with the obstacle (b).


distance is less than connection distance. In this case, the path
to goal is generated by the forward tree while the reverse tree
stops growing and initializes. The proposed rewiring method
is used to reduce the path length. To this end, the shorter
search time allows for faster generation of agent-to-goal paths,
which in turn allows for more efficient tree growth by growing
trees from points in the path to other regions. Extensive
experiments have been carried out in three different scenarios
for comparison. The results have demonstrated the validity of
our proposal, and effectively improved the motion planning
search time and path length. In particular, Bi-AM-RRT*(D)
has the best comprehensive performance, while optimizing the
search time and path length. In addition, this paper has also
discussed the influence of the value of the connection distance
on the planner and shown the practicality and robustness of
the presented approach.
It is worth noting that in the used bidirectional search
strategy, the forward tree only uses the trunk information of
the reverse tree, while its branch node information is discarded
after a successful connection. In the future work, the use of
branch information will be considered to further improve the
path optimization and obstacle avoidance performance. And
deploying our solution to mobile robots in real-world scenarios
will also be investigated in future research.


IEEE TRANSACTIONS ON INTELLIGENT VEHICLES 11



R EFERENCES


[1] L. Chen, Y. Li, C. Huang, B. Li, Y. Xing, D. Tian, L. Li, Z. Hu, X. Na,
Z. Li _et al._, “Milestones in autonomous driving and intelligent vehicles:
Survey of surveys,” _IEEE Transactions on Intelligent Vehicles_, vol. 8,
no. 2, pp. 1046–1056, 2023.

[2] X. Zhao, B. Tao, and H. Ding, “Multimobile robot cluster system for
robot machining of large-scale workpieces,” _IEEE/ASME Transactions_
_on Mechatronics_, vol. 27, no. 1, pp. 561–571, 2022.

[3] D. Zhu, B. Zhou, and S. X. Yang, “A novel algorithm of multi-auvs
task assignment and path planning based on biologically inspired neural
network map,” _IEEE Transactions on Intelligent Vehicles_, vol. 6, no. 2,
pp. 333–342, 2021.

[4] Y. Zhang, G. Tian, X. Shao, M. Zhang, and S. Liu, “Semantic grounding
for long-term autonomy of mobile robots towards dynamic object search
in home environments,” _IEEE Transactions on Industrial Electronics_,
vol. 70, no. 2, pp. 1655–1665, 2023.

[5] X. Zhang, Y. Jiang, Y. Lu, and X. Xu, “Receding-horizon reinforcement
learning approach for kinodynamic motion planning of autonomous
vehicles,” _IEEE Transactions on Intelligent Vehicles_, vol. 7, no. 3, pp.
556–568, 2022.

[6] E. W. Dijkstra, “A note on two problems in connexion with graphs,”
_Numerische Mathematik_, vol. 1, p. 269–271, 1959.

[7] P. E. Hart, N. J. Nilsson, and B. Raphael, “A formal basis for the heuristic
determination of minimum cost paths,” _IEEE transactions on Systems_
_Science and Cybernetics_, vol. 4, no. 2, pp. 100–107, 1968.

[8] M. Likhachev, G. J. Gordon, and S. Thrun, “ARA*: Anytime a* with
provable bounds on sub-optimality,” 2004, pp. 767–774.

[9] A. Stentz _et al._, “The focussed dˆ* algorithm for real-time replanning,” in
_International Joint Conference on Artificial Intelligence_, vol. 95, 1995,
pp. 1652–1659.

[10] M. Likhachev, D. I. Ferguson, G. J. Gordon, A. Stentz, and S. Thrun,
“Anytime dynamic a*: An anytime, replanning algorithm.” in _ICAPS_,
vol. 5, 2005, pp. 262–271.

[11] J. Wang, T. Li, B. Li, and M. Q.-H. Meng, “GMR-RRT*: Samplingbased path planning using gaussian mixture regression,” _IEEE Transac-_
_tions on Intelligent Vehicles_, vol. 7, no. 3, pp. 690–700, 2022.

[12] R. R. Coifman and S. Lafon, “Diffusion maps,” _Applied and computa-_
_tional harmonic analysis_, vol. 21, no. 1, pp. 5–30, 2006.

[13] Y. F. Chen, S.-Y. Liu, M. Liu, J. Miller, and J. P. How, “Motion
planning with diffusion maps,” in _IEEE/RSJ International Conference_
_on Intelligent Robots and Systems_, 2016, pp. 1423–1430.

[14] S. M. LaValle _et_ _al._, “Rapidly-exploring random trees:
A new tool for path planning,” 1998, [online] Available: https://www.cs.csustan.edu/ xliang/Courses/CS471021S/Papers/06%20RRT.pdf.

[15] J. J. Kuffner and S. M. LaValle, “RRT-connect: An efficient approach
to single-query path planning,” in _IEEE International Conference on_
_Robotics and Automation_, vol. 2, 2000, pp. 995–1001.

[16] S. Karaman, M. R. Walter, A. Perez, E. Frazzoli, and S. Teller, “Anytime
motion planning using the RRT,” in _IEEE International Conference on_
_Robotics and Automation_, 2011, pp. 1478–1483.

[17] D. Li, Q. Li, N. Cheng, and J. Song, “Extended RRT-based path planning
for flying robots in complex 3d environments with narrow passages,” in
_IEEE International Conference on Automation Science and Engineering_,
2012, pp. 1173–1178.

[18] K. Naderi, J. Rajamäki, and P. Hämäläinen, “RT-RRT*: a real-time path
planning algorithm based on RRT,” in _ACM SIGGRAPH Conference on_
_Motion in Games_, 2015, pp. 113–118.

[19] D. Armstrong and A. Jonasson, “AM-RRT*: Informed sampling-based
planning with assisting metric,” in _IEEE International Conference on_
_Robotics and Automation_, 2021, pp. 10 093–10 099.

[20] O. Khatib, “Real-time obstacle avoidance for manipulators and mobile
robots,” _The International Journal of Robotics Research_, vol. 5, no. 1,
pp. 90–98, 1986.

[21] M. Everett, Y. F. Chen, and J. P. How, “Motion planning among dynamic,
decision-making agents with deep reinforcement learning,” in _IEEE/RSJ_
_International Conference on Intelligent Robots and Systems_, 2018, pp.
3052–3059.

[22] B. Wang, Z. Liu, Q. Li, and A. Prorok, “Mobile robot path planning in
dynamic environments through globally guided reinforcement learning,”
_IEEE Robotics and Automation Letters_, vol. 5, no. 4, pp. 6932–6939,
2020.

[23] N. Pérez-Higueras, F. Caballero, and L. Merino, “Teaching robot navigation behaviors to optimal RRT planners,” _International Journal of_
_Social Robotics_, vol. 10, no. 2, pp. 235–249, 2018.




[24] J. Wang, T. Zhang, N. Ma, Z. Li, H. Ma, F. Meng, and M. Q.-H. Meng,
“A survey of learning-based robot motion planning,” _IET Cyber-Systems_
_and Robotics_, vol. 3, no. 4, pp. 302–314, 2021.

[25] Y. Kuwata, J. Teo, S. Karaman, G. Fiore, E. Frazzoli, and J. How, “Motion planning in complex environments using closed-loop prediction,” in
_AIAA Guidance, Navigation and Control Conference and Exhibit_, 2008,
p. 7166.

[26] C. Fulgenzi, A. Spalanzani, C. Laugier, and C. Tay, “Risk based motion
planning and navigation in uncertain dynamic environment,” 2010,

[online] Available: https://hal.inria.fr/inria-00526601.

[27] R. Mashayekhi, M. Y. I. Idris, M. H. Anisi, I. Ahmedy, and I. Ali,
“Informed RRT*-connect: An asymptotically optimal single-query path
planning method,” _IEEE Access_, vol. 8, pp. 19 842–19 852, 2020.

[28] J. Wang, W. Chi, C. Li, and M. Q.-H. Meng, “Efficient robot motion
planning using bidirectional-unidirectional RRT extend function,” _IEEE_
_Transactions on Automation Science and Engineering_, vol. 19, no. 3,
pp. 1859–1868, 2022.

[29] H. Ma, F. Meng, C. Ye, J. Wang, and M. Q.-H. Meng, “Bi-Risk-RRT
based efficient motion planning for mobile robots,” _IEEE Transactions_
_on Intelligent Vehicles_, vol. 7, no. 3, pp. 722–733, 2022.

[30] S. Karaman and E. Frazzoli, “Incremental sampling-based algorithms
for optimal motion planning,” _Robotics Science and Systems VI_, vol.
104, pp. 267–274, 2010.

[31] D. Yi, R. Thakker, C. Gulino, O. Salzman, and S. Srinivasa, “Generalizing informed sampling for asymptotically-optimal sampling-based kinodynamic planning via markov chain monte carlo,” in _IEEE International_
_Conference on Robotics and Automation_, 2018, pp. 7063–7070.

[32] L. Chen, Y. Shan, W. Tian, B. Li, and D. Cao, “A fast and efficient
double-tree RRT*-like sampling-based planner applying on mobile
robotic systems,” _IEEE/ASME transactions on mechatronics_, vol. 23,
no. 6, pp. 2568–2578, 2018.

[33] X. Wang, X. Li, Y. Guan, J. Song, and R. Wang, “Bidirectional potential
guided RRT* for motion planning,” _IEEE Access_, vol. 7, pp. 95 046–
95 057, 2019.

[34] F. Islam, J. Nasir, U. Malik, Y. Ayaz, and O. Hasan, “RRT*-smart: Rapid
convergence implementation of RRT* towards optimal solution,” in
_IEEE international conference on mechatronics and automation_, 2012,
pp. 1651–1656.

[35] J. D. Gammell, S. S. Srinivasa, and T. D. Barfoot, “Informed RRT*:
Optimal sampling-based path planning focused via direct sampling of an
admissible ellipsoidal heuristic,” in _IEEE/RSJ International Conference_
_on Intelligent Robots and Systems_, 2014, pp. 2997–3004.

[36] M. Owen and J. S. Provan, “A fast algorithm for computing geodesic
distances in tree space,” _IEEE/ACM Transactions on Computational_
_Biology and Bioinformatics_, vol. 8, no. 1, pp. 2–13, 2010.


