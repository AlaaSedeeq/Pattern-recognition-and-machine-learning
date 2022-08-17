# Chapter 2 Summary
    
<h3>Density estimation</h3>

It's unsupervised learninig aims to estimate a density given data from that distribution.<br>
Given N observations $(i.i.d)$ from $P(X)$, try to model the probability distribution $P(X)$.
<ul>
  <li> <b>Parametric density estimation</b> : 
      <ul>
          <li> Assumes that $P(X)$ from a known distribution goverened by some parameters w.
          <li> Try to estimate these parameters w using frequentist or bayesian:
              <ul>
                  <li> <b> Frequentist</b> : Choose w that maximize or minimize some criterion(e.g. likelihood)
                  <li> <b> Bayesian</b> : Introduce prior distributions over w, then use Bayes' theorem to esstimate the corresponding posterior given the observed data. 
              </ul>
          <li>Examples: Conjugate prior approach, Mixture models
      </ul><br>
  <li> <b>Non-parametric density estimation</b>
      <ul>
          <li> Assume that the data has a probability density function but not of a specific known form
          <li> Let data speak for themselves
          <li> Examples: Histogram, Kernel-baased, 
      </ul>
</ul>

<hr style="height:2px;">

<h3>Binary variables</h3><br>
<li> $X \in \{0, 1\}$
<li> $P(X=1)=\mu$ and $P(X=0)= 1 - \mu$ 
<li>Examples:
    <ul>
        <h4>1) Binomial distribution:</h4>
        <ul>
            <li> N trails, each with two possible outcomes.
            <li> $\text{Bin}(m|N, \mu) = {N\choose m}\mu^m(1 - \mu)^{N - m}$
            <li> Mean = $N\mu$
            <li> $\sigma^2 = N\mu(1 - \mu)$
            <li> It's conjugate prior is <b>Beta</b>
            <li> A special case, where $m = 1$ is the Bernoulli distribution:
                <ul>
                    <li> $\text{Bern}(x|\mu) = \mu^x(1 - \mu)^{1 - x}$
                    <li> $\text{Mean} = \mu$
                    <li> $\sigma^2 = \mu(1 - \mu)$
                </ul>
        </ul>
        <h4>2) Beta distribution:</h4>
        <ul>
            <li> A continuous probability distribution defined on $[0,1]$
            <li> $\text{Beta}(\mu|a,b) = \frac{\Gamma(a + b)}{\Gamma(a) \Gamma(b)} \mu^{a-1}(1 - \mu)^{b-1}$   
            <li> Suitable for modeling r.v behaviour of percentages and proportions.
        </ul>
    </ul>
        
<hr style="height:2px;">

<h3>Multinomial variables</h3><br>
    $\quad\text{Mult}(M|\mu,N) = {N\choose m{1} m{2}..m_{k}}\prod_{k=1}^K \mu_{k}^{m_{k}}, \;\;m_{k} = \sum_{n} X_{nk}$<br><br>
<li> The d-dimensional Binomial.
<li> $X$ is defined by a k-dimensional vector $X = \{x_{1},..,x_{k}\}$
<li> Only $x_{k} = 1$ and all other x are zeros
<li> The Binomial is a special case where $X \in \{x_{0}, x_{1}\}$ and one of $x_{0}$ or $x_{1}$ is one with $\mu$.
<li> It's conjugate is Dirichlet distribution:
    <ul>
        <li> The d-dimensional Beta.
        <li> $\text{Dir}(\mu|\alpha) = \frac{\Gamma(\alpha_{0})}{\Gamma(\alpha_{1})...\Gamma(\alpha_{k})}\prod_{k=1}^K \mu_{k}^{\alpha_{k}-1} ,\;\;\; \alpha_{0} = \sum^K \alpha_k$
    </ul>

<hr style="height:2px;">

<h3>Gaussian Distributions</h3><br>
<li> The sum of multiple random variables is gaussian.
<li> The fractional dependence of gaussian on X through the mahalanobis distance: <br><br>
    <p>$\quad\quad\Delta^2 = (X - \mu)^T\Sigma^{-1}(X - \mu)$<br><br>
        <li> For $N(X|\mu, \Sigma)$, $\;\; X = {X_{a}\choose X_{b}}$, $\;\;\mu = {\mu_{a}\choose \mu_{b}}$<b>:</b>
            <ul>
                <li> <b>1) The marginal distribution :</b><br><br>
                    <ul>
                        <li>$P(X_{a}) = N(X_{a}|\mu_{a}, \Sigma_{aa}^{-1})$
                    </ul><br> 
                <li> <b>2) The conditional distribution</b><br><br>
                    <ul>
                        <li>$P(X_{a}|X_{b}) = N(X_{a}|\mu_{a|b}, \; (\Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba})^{-1})$
                    </ul><br>
            </ul>
        
<li> Bayesian inference:
    <ul>
        <li> $\sigma$ is known and inferring $\mu$, the conjugate prior is Gaussian.
        <li> $\mu$ is known and inferring $\sigma$, the conjugate prior is Inverse-Gamma(Inverse-Wishart for D-dimensional).
        <li> Inferring both $\sigma$ and $\mu$, the conjugate prior is Normal-Inverse-Gamma(Normal-Inverse-Wishart for D-dimensional).
    </ul>
    
<hr style="height:2px;">
<h3>Student's t distribution</h3><br>
<p>$$\begin{aligned}
    p(x|\mu, a, b) &= \int_0^\infty \mathcal{N}\left(x |\mu, \tau^{-1}\right)\text{Gam}(\tau|a, b) \ d\tau \\
                   &= \frac{b^a}{\Gamma(a)}\left(\frac{1}{2\pi}\right)^{1/2}\left[b + \frac{(x - \mu)^2}{2}\right]^{-a - 1/2} \Gamma\left(a + \frac{1}{2}\right)
\end{aligned}$$

<p>$$\text{St}(x\vert\mu,\lambda,v) = \frac{\Gamma\big((v + 1)/2\big)}{\Gamma(v/2)}\left(\frac{\lambda}{\pi v}\right)^{1/2}\left[1 + \frac{\lambda(x - \mu)^2}{v}\right]^{-(v + 1)/2}
$$<br>

<p>Where, $v = 2a$ is the degrees of freedom of the distribution, $\lambda = a/b$ is the precision of the distribution<br><br>
    
<li> Can be defined as the marginal distribution of the unkown $\mu$ when the dependence on the unkown $\Sigma$ is marginalized out.
<li> Can be interpreted as <b>infinite mixture of Gaussian</b>  
<li> Its maximum likelihood solution can be found by expectation maximization algorithm. 
    
<hr style="height:2px;">
<h3>Periodic variable</h3><br>
<li> Used to model some continuous r.v that Gaussian can't be used(e.g wind direction at some point).
<li> To evaluate the mean of $D=\{\theta_{1},...,\theta_{N}\}$, $\theta$ id in radian

<p>$$\bar \theta = \tan^{-1}\left(\frac{\sum_n \sin(\theta_n)}{\sum_n \sin(\theta_n)}\right)\\
\bar X = \frac{1}{N}\sum^NX_{n}$$
    
<li> The periodic generalization of Gaussian is von Mises:<br>

<p>$$p(\theta|\theta_0, m) = \frac{1}{2\pi I_0(m)}\exp(m\cos(\theta - \theta_0))$$

Where $\theta_0$ is the mean, $m$ is the concentration parameter, $I_0(m)$ is the norm-coefficient which is the zeroth-order Bessel function.
