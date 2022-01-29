---
<br>
<h2 align="center"> Hunting down Asteroids using Machine Learning 📈</h2>
<h4 align="center"><img src="https://user-images.githubusercontent.com/69205415/139040664-fd5b464c-742f-4fc0-8b50-8c4c39c2aa08.jpg"  width="600" height="250"/>
</h4>

<div align="center">
 

#### Shivani Santosh Sorte
<br><br>

---
<br>
 
 ### Task : To Predict and caliber the magnitude of asteroids.
<p> 
 Asteroids are large, irregularly shaped rocky planetoids in space that orbit our Sun. Asteroids range in different sizes. If one of these giant rocks end up colliding with our planet, we might get in big trouble. Thus, we need to be able to predict and caliber the magnitude of asteroids, which is exactly what have been done in the <a href="https://github.com/shivanisorte/Machine-Learning/blob/main/Solution/ShivaniMlPraxis.ipynb
  ">attached python notebook</a> using machine learning.
</p>
 
 <br>
 
 ### Importing Dataset
<p> 
 After mounting the notebook at a proper location in drive, the first thing that we do is importing the dataset. The dataset provided to us in Praxis 2021 is a vast dataset with around 8 lakh rows and 31 columns of data related to asteroid.
</p>
 
 <br>
 
 
 ### Getting a rough understanding of the data provided
<p> 
Since our dataset is very huge, it is evident that there's a lot of unneeded data that needs to be dealt with properly in order to get good model accuracy. To be able to do that properly, let's get a rough understanding of the data provided. 
 </p>
 <img scr="https://user-images.githubusercontent.com/69205415/139370884-d25f66d6-d878-4482-a432-dc7c4e680ffe.png">
 
<p>
 We have 839714 rows and 31 columns. Let's check the first and last five data entries.
</p>
 
 <img src="https://user-images.githubusercontent.com/69205415/139371328-c0e8fedd-3750-44ae-853f-e9bf08cd1cb7.png">
 <br>
<img src="https://user-images.githubusercontent.com/69205415/139371142-8aa88510-3138-4296-bc00-9ad2c0e51179.png">


 <p>
 On getting a small peek, we understand that there are quite some categorical entries that need to be encoded properly, inconsintent diameter values (some float, some int) that need to be made consistent, and a lot of NaNs in the dataset that need to be dealt with after properly understanding all variables. Next, we also find that the diameter is in string format.
  
  
  <img src="https://user-images.githubusercontent.com/69205415/139377378-a996947d-6d80-4eb0-a308-ef4094f84aad.png">

 </p>

 
 <br>
 
  ### Converting diameter to float
<p> 
 Since the diameter is in string format and has inconsistent entries, we need to convert it to float datatype for further processing
 
</p>
 

 <img src="https://user-images.githubusercontent.com/69205415/139378373-3366cffd-c56c-43e1-972c-77b2324a10de.png">
 
 
  ### Understanding variables and Choosing a Target parameter

| Variable | Meaning |
|----|----|
 | Neme | Asteroid's name |
| a | semi-major axis[au] |
| e | eccentricity- determines the amount by which its orbit around another body deviates from a perfect circle|
| i | inclination wrt x-y ecliptic plane [deg] |
| om |  longitude of the ascending node (angle from a specified reference direction, called the origin of longitude, to the direction of the ascending node) |
| w | argument of perihelion - angle from the body's ascending node to its periapsis |
| q | perihelion (closest to sun) distance [au] - P=a(1−e)  |
| ad | aphelion (farthest from sun) distance [au] - A=a(1+e) |
| per_y | orbital period [years] |
| data_arc | data arc-span [d] (time span between earliest and latest observation) |
| condition_code | orbit condition code |
| n_obs_use | number of observations used |
| H | absolute magnitude parameter (apparent magnitude that the object would have if it were viewed from a distance of exactly 10 parsecs ) |
| neo | near earth object |
| pha | physically hazardous asteroid |
| diameter | diameter of asteroid [km] |
| extent | object bi or tri-axial ellipsoid dimensions [km] |
| albedo | geometric albedo (the ratio of actual brightness as seen from the light source to that of an idealized flat, fully reflecting, diffusively scattering disk with the same cross-section.) |
| rot_per | rotational period |
|mg| (standard gravitational parameter) product of Gravitational constant and asteroid's mass (<a href="https://meetingorganizer.copernicus.org/EPSC-DPS2019/EPSC-DPS2019-1485-3.pdf">Referece</a>) |
| bv | color index B-V magnitude difference-smaller, blue, hot |
| ub | color index U-B magnitude difference |
| IR |color index I-R magnitude difference |
| spec_B | spectral taxonomic type (SMASSII) |
| spec_T | spectral taxonomic type (Tholen) |
| G | Magnitude slope parameter |
| moid | earth minimum orbit intersection distance [au]  (the distance between the closest points of the osculating orbits of two bodies) |
| class | classes of asteroid |
 |n |  rotation axis orientation (<a href="https://issfd.org/ISSFD_1999/pdf/ODY_4.pdf">Reference</a>)  |
 
 <br>
 
   #### TARGET PARAMETER - DIAMETER
<p> 
 Our main task is to predict and caliber the magnitude of asteroids and comets. To do that, we have to choose a target parameter, for the estimation of which, we'll test various ML Models. After going through the data provided and understanding the task and the significance of each variable, it is understood that the diameter poses as one of the most significant parameters. The magnitude of the diameter demonstrates the magnitude of the asteroids. It is also directly related to many other parameters like the semi-major axis (a), absolute magnitude (H), albedo, and more. Moreover, it can also be used to tell if an asteroid is a potentially hazardous one or not.

 Thus, the target parameter chosen - <b>Diameter</b>.
</p>
 
 
   ### Preparing the dataset
 This includes cleaning and transforming raw data into useful information for further analysis and processing.
 <br>
 
 #### HANDLING MISSING VALUES
 <br>
 There's a large difference between the min and max value, and the mean or meadian don't do justice to all values of diameter. Thus, we will be dropping values instead of replacing them with the mean or median.
 
  ![image](https://user-images.githubusercontent.com/69205415/139461429-682b1df5-3fce-4261-95f9-1bdd3a0c37a3.png)
 
 
 <p>We have too many NaN values in the dataset. On calculating, we find that 37.68% of the data is missing</p>
 <img src="https://user-images.githubusercontent.com/69205415/139386748-8978e8e8-1fab-4dd1-8abe-ee77238a704b.png">
 <br>
 <img sre= "https://user-images.githubusercontent.com/69205415/139393257-bf59c7a2-0981-475e-a98c-cb3a0aac4152.png">
 <br>
 
 <p>If we drop all NaN values, we'll lose all data.</p>
 <img src="https://user-images.githubusercontent.com/69205415/139388229-9255c5ce-0c83-4c0c-ba62-a9cb4e79231e.png">

If we drop columns with more than one missing value, we lose all important data.

![image](https://user-images.githubusercontent.com/69205415/139461200-c95203d5-01d4-4124-9c9e-245337505d5f.png)

 We know the correlation of columns with the diameter column.
 
![image](https://user-images.githubusercontent.com/69205415/139462289-dead30ca-e0f3-4a22-ad7b-d260f0e91ed6.png)

 Thus, we decide to make sure to include diameter, GM, H, data_arc, n_obs_used, moid, q, BV and n, UB and a in our model.
 
 So we drop the data from 'albedo', 'ad','i','e','per','per_y','G','ma','rot_per','w','om' and'IR'.

 Since we have only 14 non null GM values, 1021 BV values and 979 UB values, we need to check if we can substitude some values in the place of missing data or if we need to drop these columns.
 
 ![image](https://user-images.githubusercontent.com/69205415/139462722-1b1aa393-f98f-48ad-be76-6b4a87666c32.png)
![image](https://user-images.githubusercontent.com/69205415/139462762-c28abad8-4ac9-4b40-9cbd-ddeff4bebf41.png)

 The data looks quite uniform, so we fill the missing values of these columns with the mean. After that, we drop non numeric columns features like name, condition_code, neo and more because the asteroid's name or, where the asteroid lies or, whether it is near earth or not or, the taxonomy has nothing to do with the diameter. 
 Then we drop records with na values and obtain clean data with 136760 rows × 12 columns.
 
 ![image](https://user-images.githubusercontent.com/69205415/139463244-35313312-63b6-4af4-9ceb-0afdc7333ecd.png)

#### CATEGORICAL ENCODING
 
 The pha values are characters 'N' and 'Y' representing categorical data. Next, we convert pha to int datatype and perform categorical encoding.
 
 <b>We have finally obtained a clean and consistent dataset </b>
 
 ![image](https://user-images.githubusercontent.com/69205415/139463606-c8addec0-cd4a-4c4d-83ee-fce628548430.png)

 #### CHECKING CORRELATION HEATMAP AND DROPPING COLUMNS WITH MULTICOLLINEARITY PROBLEMS
 
 ![image](https://user-images.githubusercontent.com/69205415/139463806-be3123e7-6d53-4d49-b48e-594c10495c30.png)

 Some multicollinearity problem in the 'data_arc' and 'n_obs_used' columns is observed. So, I dropped 'n_obs_used' first and checked how well the models were fitting, 'data_arc' next, and lastly both, 'n_obs_used' and 'data_arc'.
 
On dropping 'data_arc', we will obtain,
 
 ![image](https://user-images.githubusercontent.com/69205415/139799963-84d3367a-fdc3-4d44-9c98-1013b36e7373.png)

 
 On dropping 'n_obs_used', we will obtain,
 
 ![image](https://user-images.githubusercontent.com/69205415/139800059-b81d78bf-21bb-48af-a87b-9be4fc587f4e.png)

 
 On dropping both, 'data_arc' and 'n_obs_used', we will get,
 
 ![image](https://user-images.githubusercontent.com/69205415/139800134-7814b8dd-8540-4d54-a04b-bd119472ed7f.png)

 
 We see that on dropping 'n_obs_used', we get better results, so we continue with that.
 
 <b> The data preparation step has been completed. </b>
 
 
 ### Splitting data into model features and the target
 
 We split data into target (diameter) and features (other parameters).
 
 ### FEATURE SCALING
 Now, we tranform our input data so that it fits within a specific scale. We use the standardization method using a class from sklearn.
 
 ### TRAIN TEST SPLIT
 
 Next, we split our already splitted data (into features and target) into two parts- train dataset and test dataset.
 
 ### TRYING DIFFERENT ML ALGORTHMS
 
 We will try 14 different Machine Learning Algorithms to find out which gives the best score.
 Following are the ML algorithms with their root mean square errors and R2 scores. R2 score is a statistical measure that represents the goodness of fit of a regression model. The ideal value for it is 1. The closer it is to 1, the better is the model fitted. Root mean square error shows the standard deviation from the ideal results.
 
 
 #### 1. LINEAR REGRESSION
Root mean square error : 6.612885651364122<br>
R2 Score :  0.5922397256301632

 ![image](https://user-images.githubusercontent.com/69205415/139801697-682b4fee-1362-4640-95b6-40810163667a.png)

 
 #### 2. DECISION TREE
Root mean square error : 1.5005977963853847<br>
R2 Score :  0.9790032781306461
 
 ![image](https://user-images.githubusercontent.com/69205415/139801781-e32375a7-1fce-4fe0-ab0d-8abc4d637692.png)

 
 
 #### 3. RANDOM FOREST
Root mean square error : 2.859293390065279<br>
R2 Score :  0.9237674710590428

 ![image](https://user-images.githubusercontent.com/69205415/139801841-7281489a-2ec0-4926-a7cb-2bfd9cd8c7a8.png)

 
 
 #### 4. Elastic Net CV
Root mean square error : 7.204652576992108<br>
R2 Score :  0.5159959754061155

 ![image](https://user-images.githubusercontent.com/69205415/139801909-a3a01cea-4ea2-4585-bc30-c545d3038fb2.png)

 
 #### 5. K-Nearest Neighbours
Root mean square error : 5.108410921891511<br>
R2 Score :  0.7566706274555269
 
 ![image](https://user-images.githubusercontent.com/69205415/139802044-3b3dfc31-350c-47cd-ac1a-c4c163601052.png)

 
 #### 6. RIDGE
Root mean square error : 6.612954434828743<br>
R2 Score :  0.5922312430064647
 
 ![image](https://user-images.githubusercontent.com/69205415/139802119-fef5fad0-613c-42a2-bb2e-9ffad46b42ad.png)

 
#### 7. MLP Regression
Root mean square error : 8.54729177330212<br>
R2 Score :  0.31879167688744114
 
 ![image](https://user-images.githubusercontent.com/69205415/139802204-eeafc0fa-0634-40de-b81d-abd8ac2615a1.png)

 #### 8. Lasso
Root mean square error : 7.222946833839137<br>
R2 Score :  0.5135348613866848

 ![image](https://user-images.githubusercontent.com/69205415/139802258-236ead4f-20c1-4699-88a0-a57ca6bd3d0e.png)

 #### 9. LGBM Regression
Root mean square error : 4.508878987940813<br>
R2 Score :  0.8104341747179395

 ![image](https://user-images.githubusercontent.com/69205415/139802345-979c88cd-a7e9-40e8-b7a4-da8d4aa41a61.png)

 
 #### 10. XGBoost
Root mean square error : 2.57801205958267<br>
R2 Score :  0.9380283904567966

 ![image](https://user-images.githubusercontent.com/69205415/139802393-75221d46-01eb-47f9-ace2-9a1b647d73c2.png)
 
 
#### 11. CatBoost Regression
Root mean square error : 2.719018577253717<br>
R2 Score :  0.9310638164310532

 
 ![image](https://user-images.githubusercontent.com/69205415/139802489-3b788e34-d3ca-457a-a2b5-faf164e8c62d.png)

 
 
 #### 12. Bayesian Ridge
Root mean square error : 6.613354573624478<br>
R2 Score :  0.5921818947046238
 
 ![image](https://user-images.githubusercontent.com/69205415/139802560-40e75919-e70b-4a4a-bc45-435fdacaab4a.png)

 
#### 13. Gradient Boosting
Root mean square error : 2.2915169442224825<br>
R2 Score :  0.9510368857885992
 
![image](https://user-images.githubusercontent.com/69205415/139802655-c2c9649b-a37f-4dcd-9069-25a4ca06d2d3.png)

 
 
 #### 14. Support Vector Machine
Despite being the slowest, this model shows very poor results.
Root mean square error : 8.036758493457786<br>
R2 Score :  0.3977390255895612
 
 ![image](https://user-images.githubusercontent.com/69205415/139802740-6081153b-80ec-444c-bf07-fbdb801e6e57.png)

 
 ### CONCLUSION
 
 This is a summary of how well the models performed.

 
![image](https://user-images.githubusercontent.com/69205415/139804000-135a02e7-b4d8-4e30-955b-23d0e47b5a03.png)

 
 
 
 
 <b>Decision Tree Regression</b> with the score of <b>0.9790032781</b> shows the best performance.
 </div>
