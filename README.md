神经网络 实验1
===============================

学号：MF20330086  
姓名：王子翊  
  
代码文件结构简介：  
  
```
code     
├── README.md           : 本文件       
├── WZYNet              : WZYNet库实现      
│   ├── __init__.py      
│   ├── dataloader.py   : 实现一个简单的DataLoader（针对本实验）      
│   ├── loss.py         : 实现一些简单的损失函数：L1，MSE ✓      
│   ├── module.py       : 实现一个简单的Linear层的前向反向传播      
│   ├── optim.py        : 实现一些简单的优化器：SGD，Adam。优化参数须注册到优化器中。      
│   ├── parameter.py    : 实现对参数的包装    
│   └── utils.py        : 实现3D绘图函数       
├── np_solution.py      : 实验主文件      
├── pytorch_solution.py : *模型验证代码*       
├── test.py         : *单元测试主文件*      
└── testWZYNet          : *单元测试代码*      
    ├── __init__.py        
    ├── test_dataloader.py        
    ├── test_loss.py                
    ├── test_module.py    
    └── test_utils.py 
```
