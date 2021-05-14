# Casida-realtime equivalence

We want to find the relationship between the exponential damping used in the
real-time calculations and the Lorentzian function used in the python
script. Both of these functions, are used to to give a certain broadening to
the calculated excitations. We will try to find the relationship between
$\tau$ and $\Gamma$.

$$ \large
\displaystyle
damp(t)=e^{(-\frac{t}{\tau})} \quad \text{\&} \quad L(x) = \frac{1}{\pi} \frac{\frac{1}{2}\Gamma}{(x-x_0)^2+(\frac{1}{2}\Gamma)^2}
$$

The fourier transform of the exponential function $e^{-2\pi k_0 |x|}$ would be:

$$ \large
\displaystyle
\begin{aligned}  
 F_x[e^{-2\pi k_0 |x|}](k) &= \int_{-\infty}^{\infty} e^{-2\pi k_0 |x|} e^{-2\pi ikx} dx \\
\end{aligned}
$$

Partitioning the integral in the two parts of the module and then, using the
Euler's formula $e^{i\theta}=\cos(\theta)-i\sin(\theta)$, we got:

$$ \large
\displaystyle
\begin{aligned}  
 F_x[e^{-2\pi k_0 |x|}](k) =& \int_{-\infty}^{\infty} e^{-2\pi k_0 |x|} e^{-2\pi ikx} dx \\
 =& \int_{-\infty}^{0} e^{2\pi k_0 x} e^{-2\pi ikx} dx
 + \int_{0}^{\infty} e^{-2\pi k_0 x} e^{-2\pi ikx} dx \\
 =& \int_{-\infty}^{0} [\cos(2\pi kx)-i\sin(2\pi kx)] e^{2\pi k_0 x} dx
  \\ &+\int_{0}^{\infty} [\cos(2\pi kx)-i\sin(2\pi kx)] e^{2\pi k_0 x} dx \\
\end{aligned}
$$

Now, we propose a variable change using $u=-x$ (so $du=-dx$). Using the new
variable and the properties of trigonometric functions:

$$ \large
\displaystyle
\begin{aligned}  
 F_x[e^{-2\pi k_0 |x|}](k) 
 =& \int_{\infty}^{0} [\cos(2\pi ku) +i\sin(2\pi ku)] e^{-2\pi k_0 u} (-du)
  \\ &+\int_{0}^{-\infty} [\cos(2\pi ku) +i\sin(2\pi ku)] e^{2\pi k_0 x} (-du) \\
  \text{inverting limits:}  & \\
 =& \int_{0}^{\infty} [\cos(2\pi ku) +i\sin(2\pi ku)] e^{-2\pi k_0 u} du
  \\ &+\int_{-\infty}^{0} [\cos(2\pi ku) +i\sin(2\pi ku)] e^{2\pi k_0 x} du \\
  \text{by simetry:}  & \\
 =& \int_{0}^{\infty} [\cos(2\pi ku) +i\sin(2\pi ku)] e^{-2\pi k_0 u} du
  \\ &+\int_{0}^{\infty} [\cos(2\pi ku) -i\sin(2\pi ku)] e^{-2\pi k_0 x} du \\
  \text{simplyfying:}  & \\
F_x[e^{-2\pi k_0 |x|}](k) =& 2\int_{0}^{\infty} \cos(2\pi ku) e^{-2\pi k_0 u} du \\
\end{aligned}
$$

The integral argument of the right side of the equation represents an
exponential damped cosene function and can be resolved by double integration
by parts (use always the exponential as $dv$). So we finally got:

$$ \Large
\displaystyle
 F_x[e^{-2\pi k_0 |x|}](k) = \frac{1}{\pi} \frac{k_0}{k^2+k_0^2}
$$

which is the Lorentzian function. This last equation represents our
relationship between the damping parameter $\tau$ of the real time
calculation ($k_0$ inside the exponential) and the damping from the
Lorentian spectrum $\Gamma$ from Casida (related to the wide parameter $k_0$
in the Lorentzian function).

So, from the left-side of the last equation we got the equivalence for real-time:

$$ \Large 
\displaystyle
\begin{aligned}
  -2\pi k_0 =& -\frac{1}{\tau} \\
  k_0 =& \frac{1}{2\pi \tau}
\end{aligned}
$$

From the right-side we got the equivalence for the Lorentzian function (Casida):

$$ \Large 
\displaystyle
\begin{aligned}
  k_0 =& \frac{1}{2} \Gamma \\
  2k_0 =& \Gamma \\
  \text{replacing} \quad \quad & \\
  \frac{1}{\pi \tau} =& \Gamma
\end{aligned}
$$

So, in order to have the same curve shape, the $\Gamma$ parameter for the
Lorentzian function need to be related with the $\tau$ from real time as
shown in the las equation. 
As the $\tau$ in the FT for the real time spectrum is defined in fs and the
$\Gamma$ unit is eV, a **factor of $h10^{15}$.**

![spectrum.png](attachments/d1e08909.png)

### Bibliography:
[Fourier Transform--Exponential Function -- from Wolfram MathWorld](https://mathworld.wolfram.com/FourierTransformExponentialFunction.html)
[Euler's formula](https://www.math.columbia.edu/~woit/eulerformula.pdf)