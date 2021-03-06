# Bayesian Coding Python
import numpy as np
import scipy
import scipy.stats as stats
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
import seaborn as sns

# Best: Bayesian Estimation Supersedes T-test
drug = (101,100,102,104,102,97,105,105,98,101,100,123,105,103,100,95,102,106,
        109,102,82,102,100,102,102,101,102,102,103,103,97,97,103,101,97,104,
        96,103,124,101,101,100,101,101,104,100,101)
placebo = (99,101,100,101,102,100,97,101,104,101,102,102,100,105,88,101,100,
           104,100,100,100,101,102,103,97,101,101,100,101,99,101,100,100,
           101,100,99,101,100,102,99,100,99)

y1 = np.array(drug)
y2 = np.array(placebo)
print(y1.mean(), y2.mean())

# Prior
# 만약 mu_A, mu_B에 대한 사전 지식이 없다면 무정보 사전 분포를 정의하는 것이 좋음
# 1) mu_A, mu_B Prior: 정규 분포
# 2) std_A, std_B Prior: 균일 분포
# 3) nu_minus_1: 자유도 v의 분포: 이동된 지수 분포
pooled_mean = np.r_[y1, y2].mean()
pooled_std = np.r_[y1, y2].std() * 2

with pm.Model() as model:
    # Prior
    group1_mean = pm.Normal("group1_mean", mu=pooled_mean, sd=pooled_std)
    group2_mean = pm.Normal("group2_mean", mu=pooled_mean, sd=pooled_std)
    group1_std = pm.Uniform('group1_std', lower=1, upper=10)
    group2_std = pm.Uniform('group2_std', lower=1, upper=10)
    nu = pm.Exponential("nu_minus_one", 1/29) + 1

    # Likelihood: Noncentral T-distribution
    # Lambda
    lambda_1 = group1_std ** -2
    lambda_2 = group2_std ** -2

    # Likelihood
    group1 = pm.StudentT("drug", nu=nu, mu=group1_mean, lam=lambda_1, observed=y1)
    group2 = pm.StudentT("placebo", nu=nu, mu=group2_mean, lam=lambda_2, observed=y2)

# Estimation
with model:
    diff_of_means = pm.Deterministic('difference of means', group1_mean - group2_mean)
    diff_of_stds = pm.Deterministic('difference of stds', group1_std - group2_std)
    effect_size = pm.Deterministic('effect size',
                                   diff_of_means / np.sqrt((group1_std**2 + group2_std**2) / 2))

# Check Model Initialization
model.check_test_point()
# Model Graphs
pm.model_to_graphviz(model)

# pm.kdeplot(np.random.exponential(30, size=10000), fill_kwargs={'alpha': 0.5})

with model:
    # Prior
    # prior = pm.sample_prior_predictive(500)

    # MCMC
    step = pm.Metropolis()

    # Posterior
    # start = pm.find_MAP()
    trace = pm.sample(draws=20000, step=step, progressbar=True)
    burned_trace = trace[10000:]

pm.plot_posterior(burned_trace,
                  var_names=['group1_mean','group2_mean', 'group1_std', 'group2_std', 'nu_minus_one'])

pm.plot_posterior(trace, var_names=['difference of means','difference of stds', 'effect size'],
                  ref_val=0, color='#87ceeb')

pm.summary(burned_trace, var_names=['group1_mean','group2_mean'])
pm.summary(burned_trace, var_names=['difference of means','difference of stds', 'effect size'])
trace_df = pm.trace_to_dataframe(burned_trace)
mu_A_trace = trace_df["group1_mean"]
mu_B_trace = trace_df["group2_mean"]

def trace_hist(data, label, **kwargs):
    return plt.hist(data, bins=40, histtype="stepfilled", alpha=.95, label=label, **kwargs)

ax = plt.subplot(2, 1, 1)
trace_hist(mu_A_trace, "A")
trace_hist(mu_B_trace, "B")
plt.legend ()
plt.title("Posterior distributions of $\mu$")
#--------------------------------------------#
# 프로그래머를 위한 베이지안 with 파이썬
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


# 2.2. 모델링 방법
# 예제: 거짓말에 대한 알고리즘
# p = 부정행위자의 비율, 갖고 있는 정보가 없으므로 Prior로 Uniform 분포를 부여함
# N: 학생 수, X: 부정행위에 대해 "예"라고 답변한 학생의 수
# 1: 부정행위를 함, 0: 아무 짓도 하지 않음
# 동전던지기: 1: 앞면, 0: 뒷면
N = 100
X = 35
with pm.Model() as model:
    # Prior
    p = pm.Uniform("p", 0, 1)
    true_answers = pm.Bernoulli("true_answers", p, shape=N, testval=np.random.binomial(1, 0.5, N))
    first_flips = pm.Bernoulli("first_flips", 0.5, shape=N, testval=np.random.binomial(1, 0.5, N))
    # 모두가 2번째 동전을 던지는 것은 아니지만 그래도 똑같이 세팅한다.
    second_flips = pm.Bernoulli("second_flips", 0.5, shape=N, testval=np.random.binomial(1, 0.5, N))

    # 부정행위를 했다는 답변이 관측된 비율의 결과 추정치
    val = first_flips*true_answers + (1-first_flips)*second_flips
    observed_proportion = pm.Deterministic(name="observed_proportion",
                                           var=tt.sum(val)/float(N), model=model)
    print("observed_proportion: ", np.round(observed_proportion.tag.test_value, 2))

    # Likelihood
    observations = pm.Binomial("observarions", N, observed_proportion, observed=X)

    # MCMC
    step = pm.Metropolis(vars=[p])
    trace = pm.sample(40000, step=step)
    burned_trace = trace[15000:]

# 시각화
p_trace = burned_trace["freq_cheating"][15000:]
plt.hist(p_trace, histtype="stepfilled", normed=True, alpha=0.85, bins=30,
         label="posterior distribution", color="#348ABD")
plt.vlines([.05, .35], [0, 0], [5, 5], alpha=0.3)
plt.xlim(0, 1)
plt.legend()


# PyMC 대안 모델




