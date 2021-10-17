# MPCR-Net: Multiple partial point clouds registration network using a global template
 Point cloud registration is a key step in 3D measurement, and its registration accuracy directly affects the accuracy of 3D measurements. In this study, we designed a novel MPCR-Net for mul-tiple partial point cloud registration networks. First, an ideal point cloud was extracted from the ideal CAD model of the workpiece and was used as the global template. Next, a deep neural network was used to search for the corresponding point groups between each partial point cloud and the global template point cloud. Then, the rigid body transformation matrix was learned ac-cording to these correspondence point groups to realize the registration of each partial point cloud. Finally, the iterative closest point algorithm was used to optimize the registration results to obtain a final point cloud model of the workpiece. We conducted point cloud registration exper-iments on untrained models and actual workpieces, and by comparing them with existing point cloud registration methods, under a certain initial position deviation of the point cloud to be registered, we verified that the MPCR-Net could improve the accuracy and robustness of the 3D point cloud registration.

	1. Experiments based on untrained models
	
We used deep learning-based algorithms such as MPCR-Net, PRNet, and RPM-Net, which have correspondence point estimation and point cloud local registration functions to carry out the correspondence point estimation experiments.

![Image text](https://github.com/sukong19/pictures/blob/main/1.6.png)        ![Image text](https://github.com/sukong19/pictures/blob/main/2.1.png)     
--------------------------------  (a)  -------------------------------    -------------------------(b)         
Figure 1. Average registration error with different proportions of correspondence points: (a) Average calculation error of rotation angle; (b) Average calculation error of translation distance

Figure 1 shows that average calculation errors of the rotation angle and the transla-tion distance are negatively correlated with the proportion of correspondence points, indi-cating that the point cloud registration accuracy decreases as the proportion of corre-spondence points decreases. This is because the coordinate information of the corre-spondence points directly participates in the calculation of the rigid body transformation matrix in the point cloud registration. Since the average estimation accuracy of the corre-spondence points of PRNet and RPM-Net is significantly lower than that of MPCR-Net, the average registration accuracy is also lower than that of MPCR-Net.

	2. Experiments with actual workpieces

We used the MPCR-Net to perform a point cloud registration experiment on actual workpieces and then generated the surface reconstruction models. Then, we evaluated the point cloud registration accuracy by detecting the deviations between surface reconstruc-tion models and actual digital models. Finally, we compared with other point cloud regis-tration algorithms, such as PR-Net and RPM-Net, to verify the effectiveness and advance-ment of MPCR-Net.

![Image text](https://github.com/sukong19/pictures/blob/main/3.1.png)

Figure 3. Partial point clouds of the hull  at different scanning angles
 
 ![Image text](https://github.com/sukong19/pictures/blob/main/4.1.png) ![Image text](https://github.com/sukong19/pictures/blob/main/5.3.png)

---------------------  (a)  ------------------------------------------(b)   

Figure 4. The final 3D reconstruction results: (a) Full registered point cloud; (b) Surface recon-struction model.
 
 Take the Partial point clouds of the hull(Figure 3) as an example, use MPCR-Net to register all the partial point clouds to the global template point cloud, and use the ICP algorithm to opti-mize the registration results, then generate the full registered point cloud. Figure 18 shows the full registration point cloud (Figure 4(a)) and its surface reconstruction model (Figure 4(b)) of hull.
 

![Image text](https://github.com/sukong19/pictures/blob/main/6.1.png)

Figure 6. Cloud maps of contour deviation of hulls a-i calculated by MPCR-Net.

The above method is used to generate surface reconstruction models of all egg-shaped pressure hulls a-i, and the registration accuracies can be indicated by detecting the surface contour deviation between surface reconstruction models and actual digital models. The contour deviation cloud maps of hulls a-i are shown in Figure 6; the smaller the deviation, the higher the registration accuracy.

![Image text](https://github.com/sukong19/pictures/blob/main/7.1.png)

Figure 7. Registration accuracy of MPCR-Net, PRNet and RPM-Net on hulls a-i.

The maximum positive deviation, maximum negative deviation and the area ratio exceeding the distance tolerance   of the surface contour between the surface recon-struction model and the actual digital model are denoted as indicators A, B and C, respec-tively.

MPCR-Net, PRNet and RPM-Net algorithms were used to perform surface reconstruc-tion experiments on all egg-shaped pressure hulls a-i, and indicators A, B, and C were used to evaluate the reconstruction accuracy of three algorithms. The results are presented in Figure 7.

 
 Experiment results demonstrated that MPCR-Net has the following advantages:
1.	Using a global-template-based multiple partial point cloud registration method can fully guarantee the overlap rate between each partial point cloud and its correspond-ing partial template point cloud, thereby reducing the registration error and improv-ing the point cloud reconstruction accuracy.
2.	Searching for correspondence points between partial point clouds and the global template point cloud through TPCC-Net, does not require separate training for specif-ic local data of point clouds, thereby effectively reducing the correspondence point es-timation error.
3.	The rigid body transformation matrix parameters in the registration are estimated through TMPE-Net, and estimation results are robust to changes in data points. It eliminates the shortcomings of other algorithms that cannot effectively register two point clouds with significant differences in the amount of data.



		Usage

		1. Network Architecture:

![Image text](https://github.com/sukong19/pictures/blob/main/8.1.png)

Figure 8. The network architecture of the MPCR-Net 

MPCR-Net mainly comprises TPCC-Net and TMPE-Net. TPCC-Net uses a deep neu-ral network to extract and merge the features of a partial point cloud and the global tem-plate point cloud; it then “cuts out” a correspondence partial template point cloud in the global template point cloud. TMPE-Net merges the features of partial point clouds and the correspondence partial template point clouds, then iteratively learns the optimal rigid body transformation matrix.

	2. Requirements:

Table 1. Experimental environment

![Image text](https://github.com/sukong19/pictures/blob/main/2021-10-17_160159.png)

	3. Dataset:
./learning3d/data_utils/download_data.sh

	4. Train
Train TPCCNet:
train_TPCCNet.py

Train TMPENet:
train_TMPENet.py

	5. Test
Test TPCCNet:
test_TPCCNet.py

Test TMPENet:
test_TMPENet.py

	6. Test MPCR-Net with Your Own Data
	
In the test_TMPENet.py file, you can change the template and source variables with your data and use the trained model to run the Whole code, and the running result will appear automatically.You can set the save location of the results yourself.

We would like to thank the authors of Masknet PointNet, PRNet, RPM-Net and PointNetLK for sharing their codes.

	References:
	
PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation

PPFNet: Global Context Aware Local Features for Robust 3D Point Matching

PointNetLK: Robust & Efficient Point Cloud Registration using PointNet

PCRNet: Point Cloud Registration Network using PointNet Encoding

PRNet: Self-Supervised Learning for Partial-to-Partial Registration

RPM-Net: Robust Point Matching using Learned Features

MaskNet: A Fully-Convolutional Network to Estimate Inlier Points
