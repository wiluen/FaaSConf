# FaaSConf: QoS-aware Hybrid Resources Configuration for Serverless Workflows

The  workflow composition of multiple short-lived functions has emerged as a prominent pattern in Function-as-a-Service (FaaS), exposing a considerable resources configuration challenge compared to individual independent serverless functions. This challenge unfolds in two ways. Firstly, serverless workflows  frequently encounter dynamic and concurrent user workloads, increasing the risk of QoS violations. Secondly, the performance of a function can be affected by the resource re-provision of other functions within the workflow. With the popularity of the mode of concurrent processing in one single instance, concurrency limit as a critical configuration parameter imposes restrictions on the capacity of requests per instance. In this study, we present FaaSConf, a QoS-aware hybrid resource configuration approach that uses multi-agent reinforcement learning (MARL) to configure hybrid resources, including hardware resources and concurrency, thereby ensuring end-to-end QoS while minimizing resource costs. To enhance decision-making, we employ an attention technique in MARL to capture the complex performance dependencies between functions.
We further propose a safe exploration strategy to mitigate QoS violations, resulting in a safer and efficient configuration exploration. The experimental results demonstrate that FaaSConf outperforms state-of-the-art approaches significantly. On average, it achieves a 26.5\% cost reduction while exhibiting robustness to dynamic load changes.

## start

### Environment Version
- K8S(v1.23.1)
- OpenFaaS(0.16.3)

### Install dependencies (with python 3.9)
> pip install -r requirements.txt

### Benchmark
Benchmarks are located in benchmark directory. We use Locust as the workload generator to create custom-shaped load based on Azure Function Dataset. 

### Main Code
- Attmf.py: model training and serving
- RLenv.py: environment for hybrid resources configuration tuning
- updateOenpfaasYaml.sh and yaml diretory: interact with local serverless enviroments
