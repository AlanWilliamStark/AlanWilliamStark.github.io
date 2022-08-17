# Formulation of an optimal control problem. 

![image-20220619135028269](Optimal%20Control.assets/image-20220619135028269.png)



$\dot x=f(x)+g(x)u$--系统动态方程 the dynamics of system 

$x\in X \sub \R^n$--状态 state

$u(t)\in U$ -- 控制 control

J---成本函数 cost function

$f_0(x(t),u(t))$--瞬时支付 the instantaneous cost

$F_0(x)$ --终端支付 terminal cost



# Growth functions. 





# Utility functions, decreasing marginal return property. 

![image-20220621205010078](My%20Optimal%20Control.assets/image-20220621205010078.png)

![image-20220619232531639](Optimal%20Control.assets/image-20220619232531639.png)

## [Utility](https://www.investopedia.com/terms/u/utility.asp)

## [Utility function](https://www.investopedia.com/ask/answers/072915/what-utility-function-and-how-it-calculated.asp)

## [Law of Diminishing Marginal Utility](https://www.investopedia.com/terms/l/lawofdiminishingutility.asp)

# Clasification of economic optimal control problems. Cost functions. 

1. If $F_0 \neq 0, f_0 \neq 0$ Bolza problem![image-20220619215801564](Optimal%20Control.assets/image-20220619215801564.png)

2. If $f_0 \equiv 0, F_0 \neq 0$ Mayer problem

   $J(u,T) = F_0(x(T))$

3. If $F_0 \equiv 0, f_0 \neq 0$ Lagrange problem![image-20220619220052286](Optimal%20Control.assets/image-20220619220052286.png)

# Pontryagin’s maximum principle with fixed time and fixed endpoints

![image-20220621114617187](Optimal%20Control.assets/image-20220621114617187.png)

**Hamiltonian function**

![image-20220621114703150](Optimal%20Control.assets/image-20220621114703150.png)

**Optimal control**

The optimal control has to maximize the Hamiltonian function:

![image-20220621114809210](Optimal%20Control.assets/image-20220621114809210.png)

The optimal control $u^*$ is a function of the current values of the state and the adjoint variables:

![image-20220621114947179](Optimal%20Control.assets/image-20220621114947179.png)

**The optimal control** is found from **the first order extremality condition**:

![image-20220621115023620](Optimal%20Control.assets/image-20220621115023620.png)

![image-20220621115251560](Optimal%20Control.assets/image-20220621115251560.png)

**if second derivatvie < 0,** we can say that here is maximum Hamiltonian function.

**Maximized Hamiltonian**. Sometimes we mght need to use a different version of the Hamiltonian function, called the maximized Hamiltonian:

![image-20220621115953579](Optimal%20Control.assets/image-20220621115953579.png)

![image-20220621120020451](Optimal%20Control.assets/image-20220621120020451.png)

![image-20220621120204080](Optimal%20Control.assets/image-20220621120204080.png)

# Pontryagin’s maximum principle with fixed time and varying endpoints.



**Hamiltonian function**

![image-20220621114703150](Optimal%20Control.assets/image-20220621114703150.png)

**Optimal control**

The optimal control has to maximize the Hamiltonian function:

![image-20220621114809210](Optimal%20Control.assets/image-20220621114809210.png)

The optimal control $u^*$ is a function of the current values of the state and the adjoint variables:

![image-20220621114947179](Optimal%20Control.assets/image-20220621114947179.png)

**The optimal control** is found from **the first order extremality condition**:

![image-20220621115023620](Optimal%20Control.assets/image-20220621115023620.png)

![image-20220621115251560](Optimal%20Control.assets/image-20220621115251560.png)

**if second derivatvie < 0,** we can say that here is maximum Hamiltonian function.

**Maximized Hamiltonian**. Sometimes we mght need to use a different version of the Hamiltonian function, called the maximized Hamiltonian:

![image-20220621115953579](Optimal%20Control.assets/image-20220621115953579.png)

![image-20220621120020451](Optimal%20Control.assets/image-20220621120020451.png)



![image-20220621120628590](Optimal%20Control.assets/image-20220621120628590.png)

![image-20220621120637830](Optimal%20Control.assets/image-20220621120637830.png)







# Pontryagin’s maximum principle with free time

**Hamiltonian function**

![image-20220621114703150](Optimal%20Control.assets/image-20220621114703150.png)

**Optimal control**

The optimal control has to maximize the Hamiltonian function:

![image-20220621114809210](Optimal%20Control.assets/image-20220621114809210.png)

The optimal control $u^*$ is a function of the current values of the state and the adjoint variables:

![image-20220621114947179](Optimal%20Control.assets/image-20220621114947179.png)

**The optimal control** is found from **the first order extremality condition**:

![image-20220621115023620](Optimal%20Control.assets/image-20220621115023620.png)

![image-20220621115251560](Optimal%20Control.assets/image-20220621115251560.png)

**if second derivatvie < 0,** we can say that here is maximum Hamiltonian function.

**Maximized Hamiltonian**. Sometimes we mght need to use a different version of the Hamiltonian function, called the maximized Hamiltonian:

![image-20220621115953579](Optimal%20Control.assets/image-20220621115953579.png)

![image-20220621120020451](Optimal%20Control.assets/image-20220621120020451.png)

![image-20220621120908801](Optimal%20Control.assets/image-20220621120908801.png)

# Optimal control on infinite horizon. The role of discounting.   Pontryagin’s maximum principle on infinite horizon. Current value of the Hamiltonian.

![image-20220621124802310](Optimal%20Control.assets/image-20220621124802310.png)

![image-20220621125736081](Optimal%20Control.assets/image-20220621125736081.png)

![image-20220621125748290](Optimal%20Control.assets/image-20220621125748290.png)

![image-20220621125757128](Optimal%20Control.assets/image-20220621125757128.png)

![image-20220621125805432](Optimal%20Control.assets/image-20220621125805432.png)

# Dynamic programming. Its differences from PMP

![image-20220621130346728](Optimal%20Control.assets/image-20220621130346728.png)

# Derivation of the HJB equation.

![image-20220621130829607](Optimal%20Control.assets/image-20220621130829607.png)

![image-20220621130850640](Optimal%20Control.assets/image-20220621130850640.png)

![image-20220621130900771](Optimal%20Control.assets/image-20220621130900771.png)

![image-20220621130909348](Optimal%20Control.assets/image-20220621130909348.png)

# HJB equation for an optimal control problem with discounting.

![image-20220621131007032](Optimal%20Control.assets/image-20220621131007032.png)

![image-20220621131017548](Optimal%20Control.assets/image-20220621131017548.png)