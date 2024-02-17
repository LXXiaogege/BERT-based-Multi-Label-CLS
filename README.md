<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<br />
<div align="center">


<h3 align="center">基于BERT的文本多标签分类</h3>

  <p align="center">
    基于sem_eval_2018_task_1部分数据集，利用BERT预训练模型实现文本多分类模型微调
    <br />
    </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>目录</summary>
  <ol>
    <li>
      <a href="#介绍">介绍</a>
      <ul>
        <li><a href="#built-with">建立</a></li>
      </ul>
    </li>
    <li>
      <a href="#快速开始">快速开始</a>
      <ul>
        <li><a href="#环境要求">环境要求</a></li>
        <li><a href="#环境安装">环境安装</a></li>
      </ul>
    </li>
    <li><a href="#用例">用例</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->

## 介绍

基于BERT实现文本多标签多分类，参考Transformers的AutoModelForSequenceClassification利用Pytorch代码实现，
目的是更灵活的使用PyTorch应用到其他项目中（初心是PyTorch更稳定。。。尤其是在内网环境下）
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### 建立

* [![Python][Python.org]][Python-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->

## 快速开始

### 环境要求

* Python 3.6+
* PyTorch

### 环境安装

```sh
pip install -r requirements.txt
```

### 模型训练


```sh
python train.py
  ```
### 模型推理

```sh
python inference.py
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### 模型评估结果

#### Hamming Loss

hamming loss: 0.1361

#### Zero-One Loss

zero_one_loss: 0.7844

#### Classification Report

|  类别（class）   | 精度（precision） | 召回率（recall） | F1 分数（f1-score） | 样本数量（support） |
|:------------:|:-------------:|:-----------:|:---------------:|:-------------:|
|    anger     |     0.74      |    0.81     |      0.77       |      507      |
| anticipation |     0.41      |    0.33     |      0.36       |      200      |
|   disgust    |     0.70      |    0.76     |      0.73       |      516      |
|     fear     |     0.83      |    0.66     |      0.73       |      283      |
|     joy      |     0.82      |    0.83     |      0.83       |      507      |
|     love     |     0.61      |    0.43     |      0.51       |      136      |
|   optimism   |     0.71      |    0.62     |      0.66       |      400      |
|  pessimism   |     0.45      |    0.30     |      0.36       |      166      |
|   sadness    |     0.67      |    0.64     |      0.66       |      424      |
|   surprise   |     0.54      |    0.17     |      0.26       |      76       |
|    trust     |     0.36      |    0.06     |      0.10       |      71       |
|  micro avg   |     0.71      |    0.65     |      0.67       |
|  macro avg   |     0.62      |    0.51     |      0.54       |
| weighted avg |     0.69      |    0.65     |      0.66       |
| samples avg  |     0.71      |    0.67     |      0.65       |

#### 混淆矩阵

[[ 718 143] [ 97 410]]
[[1074 94] [ 135 65]]
[[ 687 165] [ 122 394]]
[[1047 38] [ 97 186]]
[[ 768 93] [ 85 422]]
[[1194 38] [ 77 59]]
[[ 864 104] [ 150 250]]
[[1141 61] [ 117 49]]
[[ 813 131] [ 153 271]]
[[1281 11] [ 63 13]]
[[1290 7] [ 67 4]]

<!-- USAGE EXAMPLES -->

## 用例

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos
work well in this space. You may also link to more resources.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->


[product-screenshot]: images/screenshot.png


[Python.org]: https://www.python.org/static/img/python-logo@2x.png

[Python-url]: https://www.python.org