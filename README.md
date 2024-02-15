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

   
1. 安装依赖

    ```sh
    pip install -r requirements.txt
    ```

### 模型训练

1. 执行训练脚本

    ```sh
   python train.py
    ```


<p align="right">(<a href="#readme-top">back to top</a>)</p>



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