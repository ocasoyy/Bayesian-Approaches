# 프로그래머를 위한 베이지안 with 파이썬
import numpy as np
import scipy
import scipy.stats as stats
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
import seaborn as sns

# Chapter1: 베이지안 추론의 철학
# 1.4. 컴퓨터를 사용하여 베이지안 추론하기
count_data = np.loadtxt("data/txtdata.csv")
n_count_data = len(count_data)

# Parameter
# create pymc3 variables corresponding to lambdas
# assign them to pymc3's stochastic variables
# 관련 변수들을 model 안에 모두 집어 넣음 (attribute로 call 가능)
with pm.Model() as model:
    alpha = 1.0 / count_data.mean()
    # variable that holds our txt counts
    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)

    tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data - 1)

print("Random Output:", tau.random(), tau.random())

with model:
    idx = np.arange(n_count_data)
    # switch: assign lambda1 or lambda2 as the value of lambda_
    # depending on what side of tau we are on
    # lambda1 ~ tau ~ lambda2
    lambda_ = pm.math.switch(tau > idx, lambda_1, lambda_2)

with model:
    # 데이터 확인 obs.observations
    obs = pm.Poisson('obs', lambda_, observed=count_data)

# MCMC를 이용한 학습
# lambda_1, lambda_2, tau의 사후확률분포로부터 수천 개의 확률변수를 반환한다.
# 확률변수들의 히스토그램을 그려보면 사후확률분포가 어떻게 생겼는지 볼 수 있다.
with model:
    step = pm.Metropolis()
    trace = pm.sample(draws=10000, step=step, tune=5000)

lambda_1_samples = trace['lambda_1']
lambda_2_samples = trace['lambda_2']
tau_samples = trace['tau']

# 시각화
ax = plt.subplot(311)
ax.set_autoscaley_on(False)

plt.hist(lambda_1_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $\lambda_1$", color="#A60628", density=True)
plt.legend(loc="upper left")
plt.title(r"""Posterior distributions of the variables
    $\lambda_1,\;\lambda_2,\;\tau$""")
plt.xlim([15, 30])
plt.xlabel("$\lambda_1$ value")

ax = plt.subplot(312)
ax.set_autoscaley_on(False)
plt.hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $\lambda_2$", color="#7A68A6", density=True)
plt.legend(loc="upper left")
plt.xlim([15, 30])
plt.xlabel("$\lambda_2$ value")

plt.subplot(313)
w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)
plt.hist(tau_samples, bins=n_count_data, alpha=1,
         label=r"posterior of $\tau$",
         color="#467821", weights=w, rwidth=2.)
plt.xticks(np.arange(n_count_data))

plt.legend(loc="upper left")
plt.ylim([0, .75])
plt.xlim([35, len(count_data)-20])
plt.xlabel(r"$\tau$ (in days)")
plt.ylabel("probability")


# Chapter 2. PyMC 더 알아보기
# 2.1. 서론: 부모변수, 자식변수, PyMC 변수
# PyMC 변수는 아래와 같이 with model 구문 내에서 생성해야 한다.
# 꼭 'model'이라고 이름 지을 필요는 없다. 다른 거 해도 된다.
with pm.Model() as model:
    parameter = pm.Exponential("poisson_param", 1.0)
    data_generator = pm.Poisson("data_generator", parameter)

# 이미 정의된 변수는 Model 바깥에서도 그 값을 확인할 수 있다.
print(parameter.tag.test_value)

# PyMCA 프로그래밍 변수에는 2가지 종류가 있다.
# 1) Stochastic Variable: 변수의 부모를 모두 알고 있어도 Random하다.
#  예를 들어 Poisson, Exponential, DiscreteUniform 클래스의 인스턴스가 그러하다.
# 2) Deterministic Variable: 변수의 부모를 모두 알고 있으면 Random하지 않다.

# Stochastic 변수 초기화
# 문자열은 변수의 이름을 나타내는데, 이후 분석에서 사후 분포를 retrieve 하기 위해 사용된다.
# 여러 개의 변수를 array 형태로 한 번에 만들수도 있다.
# 부모 변수 값이 주어졌다는 가정 하에 random()을 통해 난수를 생성할 수 있다.
with pm.Model() as model:
    some_variable = pm.DiscreteUniform("discrete_uni_var", 0, 4)
    betas = pm.Uniform("betas", 0, 1, shape=3)

print("Test Value: {}".format(betas.tag.test_value))
print("Random Value: {}".format(betas.random()))


# Deterministic 변수
# 아래와 같이 생성하거나 암시적으로 사칙연산을 통해서 생성할 수도 있음
deterministic_variable = pm.Deterministic(name='deterministic variable', var=betas, model=model)

# Deterministic 변수 내부에서 Stochastic 변수는 Stochastic 변수가 아니라 스칼라나 Array 처럼 작동한다.
def subtract(x, y):
    return x - y

with pm.Model() as model:
    stochastic_1 = pm.Uniform("U_1", 0, 1)
    stochastic_2 = pm.Uniform("U_2", 0, 1)

delta = pm.Deterministic(name="Delta", var=subtract(stochastic_1, stochastic_2), model=model)


# 모델에 관측값 포함시키기
# 1) 사전확률 그래프 확인하기
with pm.Model() as model:
    lambda_1 = pm.Exponential("lambda_1", 1.0)

samples = lambda_1.random(size=10000)
plt.hist(samples, bins=100, density=True, histtype="stepfilled")
plt.title("Prior distribution for $\lambda_1$")

# 2) 관측값 X를 모델에 포함시키기
# PyMC 변수들은 observed라는 Kwarg를 기본적으로 갖고 있음
# 데이터가 주어졌을 때 변수의 현재 값을 고정하고 싶을 때 사용한다.
data = np.array([10, 5])
with model:
    fixed_variable = pm.Poisson("fxd", 1, observed=data)

print(fixed_variable.tag.test_value)







#--------------------------------------------#
# Best: Bayesian Estimation Supersedes T-test
A = np.random.normal(30, 4, size=1000)
B = np.random.normal(26, 7, size=1000)

# Prior
# 1) mu_A, mu_B Prior: 정규 분포
# 2) std_A, std_B Prior: 균일 분포
# 3) nu_minus_1: 자유도 v의 분포: 이동된 지수 분포
pooled_mean = np.r_[A, B].mean()
pooled_std = np.r_[A, B].std()

# 만약 mu_A, mu_B에 대한 사전 지식이 없다면 무정보 사전 분포를 정의하는 것이 좋음
# 정규 분포의 표준 편차에 1000 같은 큰 숫자를 곱해주자.
# 표준편차 역시 Lower, Upper Bound를 크게 해주자.
tau = 1 / (1000*pooled_std**2)    # Precision Parameter

with pm.Model() as model:
    mu_A = pm.Normal("mu_A", pooled_mean, tau)
    mu_B = pm.Normal("mu_B", pooled_mean, tau)
    std_A = pm.Uniform("std_A", pooled_std/1000, 1000*pooled_std)
    std_B = pm.Uniform("std_B", pooled_std/1000, 1000*pooled_std)
    nu_minus_1 = pm.Exponential("nu_1", 1/29)

    # Likelihood: Noncentral T-distribution
    obs_A = pm.distributions.continuous.StudentT("obs_A", observed=A, nu=nu_minus_1 + 1, mu=mu_A, lam=1 / std_A ** 2)
    obs_B = pm.distributions.continuous.StudentT("obs_B", observed=B, nu=nu_minus_1 + 1, mu=mu_B, lam=1 / std_B ** 2)

    # MCMC
    step = pm.Metropolis([obs_A, obs_B, mu_A, mu_B, std_A, std_B, nu_minus_1])
    trace = pm.sample(draws=20000, step=step)
    burned_trace = trace[10000:]

trace_df = pm.trace_to_dataframe(burned_trace)

# Result
mu_A_trace = trace_df["mu_A"].values
mu_B_trace = trace_df["mu_B"].values
std_A_trace = trace_df["std_A"].values
std_B_trace = trace_df["std_A"].values #[:]: trace object => ndarray
nu_trace = trace_df["nu_1"].values + 1


def _hist(data,label,**kwargs):
    return plt.hist(data,bins=40,histtype="stepfilled",alpha=.95,label=label, **kwargs)

ax = plt.subplot(3,1,1)
_hist(mu_A_trace,"A")
_hist(mu_B_trace,"B")
plt.legend ()
plt.title("Posterior distributions of $\mu$")


