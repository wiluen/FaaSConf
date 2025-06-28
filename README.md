# FaaSConf: QoS-aware Hybrid Resources Configuration for Serverless Workflows(ASE2024)

The  workflow composition of multiple short-lived functions has emerged as a prominent pattern in Function-as-a-Service (FaaS), exposing a considerable resources configuration challenge compared to individual independent serverless functions. This challenge unfolds in two ways. Firstly, serverless workflows  frequently encounter dynamic and concurrent user workloads, increasing the risk of QoS violations. Secondly, the performance of a function can be affected by the resource re-provision of other functions within the workflow. With the popularity of the mode of concurrent processing in one single instance, concurrency limit as a critical configuration parameter imposes restrictions on the capacity of requests per instance. In this study, we present FaaSConf, a QoS-aware hybrid resource configuration approach that uses multi-agent reinforcement learning (MARL) to configure hybrid resources, including hardware resources and concurrency, thereby ensuring end-to-end QoS while minimizing resource costs. To enhance decision-making, we employ an attention technique in MARL to capture the complex performance dependencies between functions.
We further propose a safe exploration strategy to mitigate QoS violations, resulting in a safer and efficient configuration exploration. The experimental results demonstrate that FaaSConf outperforms state-of-the-art approaches significantly. On average, it achieves a 26.5\% cost reduction while exhibiting robustness to dynamic load changes.
![overview of FaasConf](https://github.com/wiluen/FaaSConf/blob/main/overview.png)
## start

### Environment Version
- K8S(v1.23.1)
- OpenFaaS(0.16.3)

### Install dependencies (with python 3.9)
> pip install -r requirements.txt

### Benchmark
Benchmarks are located in benchmark directory. We use Locust as the workload generator to create custom-shaped load based on Azure Function Dataset. 
#### ML workflows

These workflows perform various tasks, including pre-processing of input images and classification utilizing deep learning (DL) models.
##### orchestrate serverless workflows of ML workflows, including:
- Sequential(5 funcs)
- Parallel(6 funcs)
- Branch(7 funcs)
##### single functions: ML single functions
- load
- starter
- rgb
- update
-  resize
-  resnet
-  mobilenet

you can build the image and upload to your Docker hub then deploy it
> faas-cli build -f function.yml

> faas-cli deploy -f function.yml


the benchmark is modifed from paper [Enhancing Performance Modeling of Serverless Functions via Static Analysis（ICSOC22）](https://springer.longhoe.net/chapter/10.1007/978-3-031-20984-0_5#google_vignette)

#### TcktApp

Search ticket from TrainTicket, a comprehensive microservices suite. The role is to retrieve information about remaining tickets between two locations, with the tickets information being stored in MongoDB. You can install this applications according to the instructions of [Serverless version of TrainTicket](https://github.com/FudanSELab/serverless-trainticket), we omit here.

### Main Code
- Attmf.py: model training and serving
- RLenv.py: environment for hybrid resources configuration tuning
- Aquatope.py,RAMBO.py,firm.py to reproduce state-of-the-arts
- updateOenpfaasYaml.sh and yaml diretory: interact with local serverless enviroments

### Acknowledgments
We would like to express our special thanks to the open-source benchmarks, traces and codes of these papers or repositories:

- [GAT-MF: Graph Attention Mean Field for Very Large Scale Multi-Agent Reinforcement Learning](https://dl.acm.org/doi/abs/10.1145/3580305.3599359)
- [Enhancing performance modeling of serverless functions via static analysis](https://link.springer.com/chapter/10.1007/978-3-031-20984-0_5)
- [Serverless in the wild: Characterizing and optimizing the serverless workload at a large cloud provider](https://www.usenix.org/conference/atc20/presentation/shahrad)
- [Serverlss Trainticket](https://github.com/FudanSELab/serverless-trainticket)

### Citation
```
@inproceedings{wang2024faasconf,
  title={FaaSConf: QoS-aware Hybrid Resources Configuration for Serverless Workflows},
  author={Wang, Yilun and Chen, Pengfei and Dou, Hui and Zhang, Yiwen and Yu, Guangba and He, Zilong and Huang, Haiyu},
  booktitle={Proceedings of the 39th IEEE/ACM International Conference on Automated Software Engineering},
  pages={957--969},
  year={2024}
}
```
