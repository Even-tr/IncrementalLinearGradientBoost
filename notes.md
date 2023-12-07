#Notes

## loss functions:

* Deviance $-\sum_{k=1}^{K} \hat{p}_{mk}\ln{\hat{p}_{mk}}$
    * two class setting (with $\hat{p}$ being the proportion of the second class): $-\hat{p}\ln \hat{p} - (1-\hat{p})\ln (1-\hat{p})$

(ESL p. 309)

## (negative) Gradient of loss functions:
$-\partial L(y_i, f(x_i))/\partial f(x_i)$
Boosted methods fit on the gradient!

* Deviance $I(y_i = G_k) - p_k(x_i)$
    * two class setting (Let $\hat{p}$ be the probability of being class 1): $I(y_i = 1) - \hat{p}(x_i)$

(ESL p. 360)