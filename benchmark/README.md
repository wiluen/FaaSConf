# ML workflows

These workflows perform various tasks, including pre-processing of input images and classification utilizing deep learning (DL) models.
## orchestrate serverless workflows of ML workflows, including:
- Sequential(5 funcs)
- Parallel(6 funcs)
- Branch(7 funcs)
## single functions: ML single functions
- load
- starter
- rgb
- update
-  resize
-  resnet
-  mobilenet

you can build the image and upload to your Docker hub then deploy it
> faas-cli build -f functiin.yml
> faas-cli deploy -f functiin.yml


the benchmark is modifed from paper [Enhancing Performance Modeling of Serverless Functions via Static Analysis（ICSOC22）](https://springer.longhoe.net/chapter/10.1007/978-3-031-20984-0_5#google_vignette)

# TcktApp

Search ticket from TrainTicket, a comprehensive microservices suite. The role is to retrieve information about remaining tickets between two locations, with the tickets information being stored in MongoDB. You can install this applications according to the instructions of [Serverless version of TrainTicket](https://github.com/FudanSELab/serverless-trainticket), we omit here.
