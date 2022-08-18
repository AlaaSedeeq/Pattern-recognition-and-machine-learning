# Chapter 3: Linear Models For Regression (Summary)

<hr style="height:2px;">   
<li> Linear models (LM) are linear functions of the parameters $W$, and yet can be nonlinear with respect to the input variables.
<li> Many methods are nothing but a generalization to linear models, including NN and even classification methods.
<li> Given $D = \{(X_{1}, t_{1}),..,(X_{N}, t_{N})\}$, predict the value of $t$ for a new value of $x$. This can be done by:
    <ul>
        <p> <b>1)</b> Simple function $F(X)$ maps directly to $y$.
        <p> <b>2)</b> Probablistic perspective, where we model $P(t|X)$, it expresses our uncertainity about t for each value of x,              in such a way that minimize the expected value of loss function.
    </ul>
<li> LM are limited with high dimensional inputs.

<hr style="height:2px;">

<h3>Linear basis function models</h3><br>
<li> Linear models are not restricted to be linear in in features vector X, they can be linear in a transformation of it.
<li> Linear basis function models consider linear combinations of fixed <b>nonlinear</b> functions of the input variables.
$$
y(X, w) = w_{0} + \sum_{j=1}^{M-1} w_{j}\phi{j}
$$<br>
where, $\phi{j}$ is the basis function, $w_{0}$ is the bias.<br><br>
    
<li><b>linear regression</b> is the simplest linear model:<br>
$$
y(X, w) = w_{0} + w_{1}X_{1} +...+ w_{D}X_{D}
$$<br>
    
where the output y is a linear function of the parameters $W$, or the basis are identity.
<hr style="height:2px;">
    
<h3>Maximum Likelihood and Least Squares</h3><br>
<li> Maximizing the likelihood function is the same as minimizing the least squares under the assummption that:
    <ul>
        <li> <b>Linearity:</b> $\mathop{\mathbb{E}}[t|X] = W^TX$
        <li> <b>Homoscedasticity:</b> $\mathop{\mathbb{Var}}[\epsilon|X] = \beta^{-1}$
        <li> <b>Normality:</b> $\epsilon$ ~ $N(0, \beta^{-1})$
        <li> <b>Independence of the errors</b> : $\mathop{\mathbb{E}}[\epsilon_{i}\epsilon_{j}] = 0, i\neq j$
    </ul>

<li> Assume:
<ul>
    <li> $t = y(X, W) + \epsilon$, and $\epsilon ~ N(0, \beta^{-1})$
    <li> $P(t|X, W, \beta^{-1}) = N(t|y(X, w), \beta^{-1})$
</ul>

<li> So, we end up maximizing a gaussian form expression, which means that minimizing the exponent <b>(the mahalanobis distance)</b>.
    <ul>
    <p> $W_{ML} = argmax_{w} L(W)$ (MLE)
    <p> $W_{ML} = argmin_{w} RSS(W)$ (OLS)
    </ul>
    
<li> So the solution is:
    <ul>
        <li> $W_{ML} = (\Phi^T\Phi)^{-1}\Phi^Tt$ (Normal Equations).
        <li> $\frac{1}{\beta_{ML}} = \frac{1}{N}\sum_{n=1}^N\{t_{n} - W_{ML}^T\phi(x_{n})\}$, it's the precision matrix of the noise, given by the residual variance of $t$ around the linear model.
    </ul>
 
<hr style="height:2px;">

<h3>Maximum a posterior and Regularized Least Squares</h3><br>
<li> To control the over-fitting of MLE, we add a regularization term to the error function.
<li> Maximizing a posterior and Regularized Least Squares under the same assummption.
<br><ul>
        <p> $W_{MAP} = argmin_{W} Error(W) + \frac{\lambda}{2} \sum_{j=1}^M ||W_{j}||^q$
    </ul>
<li> With <b>p=1</b>, it's <b>Lasso regression</b>
<li> With <b>p=2</b>, it's <b>Ridge or weight decay regression</b>

<hr style="height:2px;">

<h3>Sequential Learning</h3><br>
<ul>
    <p> $W^{\tau+1} = W^{\tau} - \eta \nabla E_{n}$
</ul>    
<li> Unlike batch techniques that process all data at once (e.g MLE), in sequential learning or on-line learning data are considered sample-by-sample.
<li> The computation cost is less ($W_{ML} = (\Phi^T\Phi)^{-1}\Phi^Tt$ is very expensive).
    
<hr style="height:2px;">

<h3>The Bias-Variance Decomposition</h3><br>
<li> So, we can model some hypothesis $h(x)$ using $y(X, W)$ through two main approaches:
    <ul>
        <li> <b>Bayesian approach</b>: The uncertainty in our model is expressed through a posterior distribution over W.
        <li> <b>Frequentist approach</b>: Makes a point estimate of $W$ that maximizes or minimizes some criterion based on the given D, then tries to interpert the uncertainity of the estimator through the following steps:<br><br>
            <ul>
    <li> Assume that we have $M$ different $i.i.d$ training data sets each with  size $N$ and one test data set, and all of them are sufficiently large.<br><br>
        <ul>
            <p> <b>1)</b> Run the learninig algorithmfor each training $D$, and get $P(X; D)$.
            <p> <b>2)</b> Different $D$ results different functionsand different error values
        </ul>
            <br>
    <li> Then we have <b>three</b> types of errors to measure the performance of the model:<br><br>
        <ul>
            <p> <b>1) Apparent error</b>, also called training error. That's what we minimize while training for each training data set.
                <ul>
                    <p> $ \bar{err} = \frac{1}{N_{train}}\sum_{i\in D_{train}}(\hat y_{i} - y_{i})^2 = \frac{1}{N_{train}} \hat\epsilon^T\hat\epsilon = \frac{1}{N_{train}}||\epsilon||_{2}^2$
                </ul><br>
            <p> <b>2) Conditional error</b>, (conditioned on the training data set). That's what we really want to minimize.
                <ul>
                    <p> $err_{tr} = \mathop{\mathbb{E_{x_{0},y_{0}}}}(\hat{y_{0}} - y_{0})^2 = \mathop{\mathbb{E_{x_{0}}}} err_{tr}(x_{0})$
                    <p> $err_{tr} \approx \frac{1}{N_{test}}\sum_{i\in D_{test}}(\hat y_{i} - y_{i})^2$ 
                </ul><br>
            <p> <b>3) Unconditional error</b> 
                <ul>
                    <p> $err = \mathop{\mathbb{E_{tr}}}err_{tr} = \mathop{\mathbb{E_{x_{0}}}}[\sigma_{y_{0}|x_{0}}^2 + \mathop{\mathbb{E_{tr}}}(\hat{y_{0}} + \mathop{\mathbb{E_{tr}}}\hat{y_{0}})^2 + (\mathop{\mathbb{E_{tr}}}\hat{y_{0}} - \mathop{\mathbb{E_{y_{0}|x_{0}}}}{y_{0}})^2]$
                    <p> $err(x_{0}) = \sigma_{y_{0}|x_{0}}^2 + \mathop{\mathbb{Var_{tr}}}\hat{y_{0}} + Bias^2(\hat{y_{0}})$
                    <ul><br>
                        <p> <b>1)</b> $\sigma_{y_{0}|x_{0}}^2$ : The variance related to the data (<b>irreducible</b>)
                        <p> <b>2)</b> $\mathop{\mathbb{Var_{tr}}}\hat{y_{0}}$ : The variance of the modelsaround their average. Gives a sense of model sensitivity on particular choice of $D$.
                        <p> <b>3)</b> $(\mathop{\mathbb{E_{tr}}}\hat{y_{0}} - \mathop{\mathbb{E_{y_{0}|x_{0}}}}{y_{0}})^2$ : The bias of the average model. Shows how much the average model deviate from the optimal model.
                    </ul>
            </ul>
        </ul><br>
            <li> But in practical, we have only one data set $D$. And even if we have multiple D, we would better off combining them into a single $D$ which of course would reduce the over-fitting. 
            </ul>
                
<hr style="height:2px;">

<h3>Bayesian Linear Regression</h3><br>

<li> Over-fitting in MLE can be controlled by adding a regularization term to the log likelihood function.
<li> But the choice of the number and form of the basis functions is of course still important in determining the overall behaviour of the model. And this cannot be decided simply by maximizing the likelihood function, because this always leads to excessively complex models and over-fitting.
<li> Bayesian treatment avoids overfitting of MLE and leads to automatic methods of determining model complexity using the training data alone.
