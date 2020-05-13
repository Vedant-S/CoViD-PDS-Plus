# _Team_: B.I.N.A.R.Y | _Topic_: Healthcare<img src="https://media0.giphy.com/media/3HI5JZCU9BYxx2daxf/source.gif" width=70 height=60>
_________________________________________________________________________________________________________________________________

Our Website Prototype :   https://covid-pds.herokuapp.com/
<br>
Our App APK (WebView) :   https://bit.ly/covid-pds-apk

### Project Overview
----------------------------------

#### The Current Situation 
* The government has given contracts to reputed vendors for distribution of daily eatables and ration among the people who are suffering, during the lockdown, at SUBSIDIZED RATES according to their income.

* The vendors have set up their distribution points throughout the country, but are facing problems in management during distribution, as there are a few workforce available at every distribution point to cater the entire locality, which leads to violation of Social Distancing rule.

* The government has arranged quarters for the homeless people, as they are at a high risk of getting contaminated. But there is no medium at present, to connect them together due to the technological barrier. The citizens are helping the NGOs to locate such people, but at a slow rate.

#### To solve this problem, we are trying to develop application platform which bridges the gap between the distributors and the customers, using a Full fledged Web Application Architecture powered by Machine Learning Models.  

<br>

### Solution Description:
----------------------------------
```diff
! Problem Statement: 
- “Due to COVID-19, a huge number of people in our country are with no food and shelter.
-  How can you use technology to help the authorities solve this problem ?”
```
<img src="https://i.pinimg.com/originals/0b/9a/56/0b9a569deef6839153414ac47cc4e442.gif" width=150 height=100>**_Proposed Solution:_**
1. An application platform which bridges the gap between the distributors and the customers.

2. Using ML algorithms, we are calculating the INCOME INDEX of the customers for understanding their economic condition. This helps us to predict the % of subsidies to be given.

3. The application features the portal where citizens can report homeless people, the information would be immediately transferred to connected NGOs/ organizations.
<br>
<img src="https://github.com/Vedant-S/B.I.N.A.R.Y-COVID-PDS/blob/master/PPT/Diagrams/implement.jpg" width=900 height=300>
------------------------------------------------------------------------------------------------------------------------------------

<img src="https://thumbs.gfycat.com/AdorableUnripeArrowana-size_restricted.gif" width=100 height=100> **_Uniqueness of The Project:_**
1. Effective business model. No Direct or Indirect Competitors.

2. Multi language support to cater the diverse population.

3. The INCOME INDEX calculation through the Predictive Analytics can be used in other projects requiring classification according to economic conditions.

4. With the use of Real Time Analytical calculations and QR Code Checkout, the waiting time at the distribution points is lowered, maintaining the SOCIAL DISTANCING rule.

5. Migrant labours stuck in other cities, having their Aadhar Card can avail the service, which is not possible with the current PDS system.

<br>

#### Architecture Diagram:

* Application Workflow
<img src="https://github.com/Vedant-S/B.I.N.A.R.Y-COVID-PDS/blob/master/PPT/Diagrams/App%20Workflow.png" align="center">

<br>

* Web Architecture
<img src="https://github.com/Vedant-S/B.I.N.A.R.Y./blob/master/PPT/Diagrams/architecture.png?raw=true" align="center">

<hr>

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src="https://www.altoros.com/blog/wp-content/uploads/2018/04/kubeflow-to-automate-deployment-of-tensorflow-models-on-kubernetes-v14.gif" width=500 height=300>

## Technical Description:

* The integration and implementation of the source code by the developer with the necessary requirements of society is essential. It is done by the Operations Manager, with use of Docker/Kubernetes for container deployment of the website, and Jenkins for system based automation i.e CI/CD.

* An individual’s annual income results from various factors. Intuitively, it is influenced by the individual’s education level, age, gender, occupation, and etc.

* It is essential that the income take into consideration all factors to necessitate the use of the code to as a Classification Model.

* The first model used calculate the costs of sheltering, while the second model calculates required area as well as costs of housing the population (registered to the website/application).

* Several Classifier models are used to segregate and predict the people of one economic/financial class to the other, to avoid the discriminatory treatment of them as same with respect to their economic background.

* The most important step used by us is the use of Deep Learning concepts and Open Computer Vision for purposes of verification. An audio file and age classifiers are used to avoid wrongful addition of data, which can corrupt the database in terms of information gain.

* The deep learning models implemented can also be used to train or educate the people in these testing times. ASHA(Accredited Social Health Activist) workers also can be trained by using the various models, for example handwashing, hygiene, cleanliness etc.

* I have made use of Pytorch/Tensorflow-Keras deep learning libraries to enhance the linear models using Gradient Descent and Backpropagation. Further we have used the Opencv(i.e 'cv2') DNN(Deep Neural network) models for purposes of verifying against pranks while entering data.

* For Example, the youth of today can play a prank to enter their age as over 80, for both monetary and/or social benefit. But the audio sound (which classifies people above or below the age of 35) segregates them. Further the GAN (Generative Adversarial Network) age determination model will further bring the number to a close value.
<img src="https://github.com/Vedant-S/B.I.N.A.R.Y-COVID-PDS/blob/master/PPT/Diagrams/implement.jpg" width=900 height=300>
<br>

##### Technologies/Versions Used (for prototype):

* A MERN Stack Web Application using NGINX as Reverse Proxy and Load Balancer.
* Complete User Interface of the Mobile Application using Adobe XD.
* Utilizing the computation power of Spark for Predictive Analytics.
* Containerizing the Application using Docker for safe and secure deployment on Cloud Engines.
* OpenCV and Pytorch library (packages) used.
<br>
<img src="https://miro.medium.com/max/700/1*Z4L6D1RiQauGmB3TGK_wJg.gif" height=100 width=100> | <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/OpenCV_Logo_with_text_svg_version.svg/1200px-OpenCV_Logo_with_text_svg_version.svg.png" width=100 height=100>

##### Setup/Installations required to run the solution:

* Web Application Dependencies (For WebSite Prototype)
* ML Model Dependancies
* ADOBE XD (For UI Prototype)
* Physical/Virtual Android Device (For Android App)

<br>


### Team Members:
----------------------------------
```diff
! Vedant Shrivastava | vedantshrivastava466@gmail.com
+ Anmol Sinha        | 1805553@kiit.ac.in
- Avik Kundu         | 1828008@kiit.ac.in
```
[Go to Anmol's profile](https://github.com/anmol-sinha-coder/)
<br>
[Go to Avik's profile](https://github.com/Lucifergene)
