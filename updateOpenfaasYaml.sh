#!/bin/bash
function=$1
# deployfile=/home/user/serverless-trainticket/src/backend/FaaS/Part01/$1/$1.yml  
deployfile=/home/user/image-function/$1/$1.yml     
cpu=$2
memory=$3
replicas=$4
# concurrency_limit=$5
# backfile=/home/user/code/yaml/search/$1.yml         
backfile=/home/user/code/yaml/ml/$1.yml  


# 新建部署文件
cp $backfile $deployfile



# 修改资源配置
sed -i "s/cpu\: 50/cpu\: $cpu/g" $deployfile
sed -i "s/memory\: 64/memory\: $memory/g" $deployfile
# sed -i "s/max_inflight\: 100/max_inflight\: $concurrency_limit/g" $deployfile
# sed -i "s/cpu\: 50/cpu\: 1000/g" $deployfile
# sed -i "s/memory\: 64/memory\: 1024/g" $deployfile


# 部署函数操作，需要template模板
cd /home/user/code
faas-cli deploy -f $deployfile --label com.openfaas.scale.min=$replicas


