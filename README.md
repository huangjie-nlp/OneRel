# OneRel
chinese relation extract  
复现AAAI2022中的《OneRel: Joint Entity and Relation Extraction with One Module in One Step》  

## 环境  
python==3.8  
torch==1.8.1  
transformers==4.3.1

## 运行  
1.模型的超参数写在配置文件config.py中  
2.数据以dataset中的为例  
3.运行python main.py训练模型  
4.test.py可单条测试  
  
## 结果
本人用ccks2020的中文医学实体关系抽取的数据训练模型，训练100个epoch最后验证集的f1_score约为60.78%  

## 模型结果对比  
CasRel：f1=51.28%  
GPLinker：f1=59.15%  
OneRel：f1=60.78%  
