---
layout: post
title: opencv C++ 实现 Eigenface
subtitle: Each post also has a subtitle
categories: markdown
tags: [CV]
---

##### 实验原理

假设脸均为行向量

对中心化后的样本矩阵（一行为一个样本脸）进行 SVD，得到的右奇异矩阵为一个行与列个数均与图像像素数相同的正交矩阵，一行为一个特征脸，选择其对应奇异值最大的 $p$ 行作为特征脸，组成特征脸矩阵 $A$

将人脸样本 $f$ 映射特征脸空间时，则有 $y_f=fA^T$ ，其中 $y_f$ 则为 $f$ 降维后的结果

若需要进行重构，则有 $f_{re}=y_fA$ ，其中 $f_{re}$ 为重构之后的人脸，也为一个行向量

通过图像在特征脸空间的投影间的欧氏距离大小判断二者的相似程度
$$
Distance_k=||Ω−Ω_k||^2
$$
其中 $\Omega$ 与 $\Omega_k$ 分别表示两个人脸图像，本实验中 $\Omega$ 为测试集中一个人脸样本的投影结果，$\Omega_k$ 为训练集中的人脸样本投影结果，应将 $k$ 由 $0$ 遍历至 `training_set.size()` 找出使 $Distance_k$ 最小的 $k$

##### 本次实验的简单流程是

1. 读取人脸数据集，将自己的 10 张照片加入其中之后，将其分为训练集和测试集
2. 进行预处理，直方图均衡化，中心化（零均值化）
3. 训练，求出训练集的特征人脸
4. 将训练集的人脸与测试集的人脸均通过特征人脸映射到特征脸空间后，比较二者的欧氏距离，训练集中与每个测试集中的人脸欧氏距离最小的人脸被认为与测试集中对应的人脸是同一个人
5. 将一个人脸通过特征人脸映射到特征脸空间后，再通过得到的结果重构出原先的人脸

#### 实验步骤

##### 读入图片

首先读入AT&T图片文件，全部存入由 `Mat` 矩阵组成的数组 `faces` 中，同时进行眼睛对齐。实现眼睛对齐的方法即为仿射变换，读取每张图像的同时也读取其眼睛的位置，以第一张图像的眼睛位置为基准，通过仿射变换的方式将每张图像的双眼对齐。由于仿射变换需要三角形作为基准，因此我选择了每张图像两眼中点偏上 20 像素的位置作为基准，这样取相对位置可以尽可能减小仿射变换导致的人脸的大幅度倾斜与变形

```c++
int main()
{
    Mat faces[FACNUM * PICNUM];
    vector<Mat> training_set;
    vector<Mat> test_set;
    vector<Mat> reconstruct_set;
    Point eye1;
    Point eye2;
    int size = FACNUM * PICNUM;
    //read in the faces
    cout<<"Reading in the faces and preprocessing..."<<endl;
    string path = "att-face/";
    for(int i = 0; i < FACNUM; i++) {   //40 faces
        for(int j = 0; j < PICNUM; j++) {   //10 pics for each face
            string facenum = "";
            int temp;
            temp = i + 1;
            while(temp!=0) {
                char newnum = '0' + temp % 10;
                facenum = newnum + facenum;
                temp = temp / 10;
            }
            string picnum;
            if(j == 9) {
                picnum = "10";
            }
            else {
                picnum = '1' + j;
            }
            //cout<<path + "s" + facenum + "/" + picnum + ".pgm"<<endl;
            faces[i*PICNUM + j] = imread(path + "s" + facenum + "/" + picnum + ".pgm");
            // imshow("1", faces[i]);
            // waitKey(50);
            // cout<<i<<", "<<(int)j<<endl;

            //align the eyes
            string fpath = "ATT-eye-location/";
            FileStorage fs(fpath + "s" + facenum + "/" + picnum + ".json", FileStorage::READ);
            if(i == 0 && j == 0) {
                fs["centre_of_left_eye"]>>eye1;
                fs["centre_of_right_eye"]>>eye2;
            }
            else {
                Point temp1, temp2;
                fs["centre_of_left_eye"]>>temp1;
                fs["centre_of_right_eye"]>>temp2;
                Point2f src[] = {(temp1 + temp2)/2 - Point(0, 20), temp1, temp2};
                Point2f dst[] = {(eye1 + eye2)/2 - Point(0, 20), eye1, eye2};
                Mat affineMat = getAffineTransform(src, dst);
                warpAffine(faces[i*PICNUM + j], faces[i*PICNUM + j], affineMat, Size(faces[0].cols, faces[0].rows), 1, BORDER_REFLECT);
            }
        }
    }
    // cout<<faces[1].channels()<<endl;    //3 channels
    // cout<<faces[1].type()<<endl;    //type is 16
    rows = faces[0].rows;
    cols = faces[0].cols;
    // cout<<rows<<endl;   //112
    // cout<<cols<<endl;   //92
```

接下来建立 `training_set` ，`test_set` 与 `reconstruct_set`

这这三个集合均使用 C++ STL 容器 vector 来实现，以方便维护。选出每个人脸的前 5 张图像作为训练集，后 5 张图像作为测试集（包括自己的人脸图像）。使用自己的人脸图像作为重构集。由于读入的人脸图像为分辨率为 $1440\times960$ 的图像，且除人脸外图像中还存在其他的非必要元素，因此为了与给出的 AT&T 数据集中的人脸图像一致，先用 `colRange()` 与 `rowRange()` 将图像裁切至基本只包含人脸且长宽比与 AT&T 数据集中的图像长宽比相同，为 $96:112$ ，然后使用 `resize()` 将图像分辨率变为 $96\times112$ ，与 AT&T 数据集中图像分辨率一致。注意此处使用 `resize()` 缩放图像时，若使用默认参数给出的插值算法得到的图像锯齿将极为严重，这里手动选择 `INTER_AREA` 算法进行插值。这里由于没有实现检测眼睛中心位置的功能，因此为了检测自己人脸图像的眼睛位置，将处理后的图像输出为 `jpg` 格式图像，并在 windows 系统的”画图“中打开，找到每张图眼睛的中心位置并记录在 `pos.json` 文件中。完成后重新运行程序，即可从 `pos.json` 中读取眼睛位置数据，使用与上述相同的方法进行对齐，基准仍为 AT&T 数据集第一张图像中眼睛的位置。然后将自己人脸的图像前 5 张加入训练集中，后 5 张加入测试集中，10 张自己的人脸图像作为重构集。

```c++
    //construct training set
    for(int i = 0; i < 40; i++) {// use the first 25 faces, each face has its first 5 pics
        for(int j = 0; j < 5; j++)
        training_set.push_back(faces[i * PICNUM + j].clone());
    }

    //construct test set
    for(int i = 0; i < 40; i++) {// use the first 25 faces, each face has its last 5 pics
        for(int j = 5; j < 10; j++)
        test_set.push_back(faces[i * PICNUM + j].clone());
    }

    //construct reconstruct set
    FileStorage fs("pos.json", FileStorage::READ);
    for(int j = 0; j < 10; j++) {
        string path = "";
        path = path + (char)('0' + j) + ".jpg";
        Mat myface = imread(path);  // each image is cols == 1440, rows == 960
        myface = myface.colRange(398, 1042);
        myface = myface.rowRange(88, 872);
        resize(myface, myface, Size(92, 112), 0, 0, INTER_AREA);
        path = "";
        imwrite(path + (char)('0' + j) + "d.jpg", myface);
        //align the eyes
        Point temp1, temp2;
        string label_l = "", label_r = "";
        label_l = label_l + (char)('0' + j) + "l";
        label_r = label_r + (char)('0' + j) + "r";
        fs[label_l]>>temp1;
        fs[label_r]>>temp2;
        Point2f src[] = {(temp1 + temp2)/2 - Point(0, 20), temp1, temp2};
        Point2f dst[] = {(eye1 + eye2)/2 - Point(0, 20), eye1, eye2};
        Mat affineMat = getAffineTransform(src, dst);
        warpAffine(myface, myface, affineMat, Size(faces[0].cols, faces[0].rows), 1, BORDER_REFLECT);
        reconstruct_set.push_back(myface);
        if(j < 5) {
            training_set.push_back(myface);
        }
        else {
            test_set.push_back(myface);
        }
        // imshow("myface", myface);
        // waitKey(0);
    }
```

##### 训练

训练模型时先进行预处理，先将训练集中所有图像转换为灰度图，然后进行直方图均衡化。

```c++
    //Training
    cout<<"Start training"<<endl;
    FileStorage model("model.json", FileStorage::WRITE);
    //histogram equalization
    for(int i = 0; i < training_set.size(); i++) { // the first 200 pics are used as training set
        cvtColor(training_set[i], training_set[i], COLOR_RGB2GRAY);
        equalizeHist(training_set[i], training_set[i]);
        // imshow("afterhist", faces[i]);
        // waitKey(100);
    }
```

然后将训练集中的每张图像转化为一个行向量，合并为一个大矩阵 `samples`

```c++
    //combine all the training set into a big matrix, each row is a image
    Mat samples = training_set[0].reshape(0, 1);
    for(int i = 1; i < training_set.size(); i++) {
        vconcat(samples, training_set[i].reshape(0, 1), samples);
    }
```

接下来调用 `calcCovarMatrix()` 函数计算出矩阵 `samples` 的均值与协方差矩阵。这里我一开始使用的方法是求出协方差矩阵的特征向量作为特征人脸，但是这样做程序的运算量很大，需要运行较久才能获得结果。而现在改为使用 SVD 来求取特征人脸，协方差矩阵已经用不到了，但是 `calcCovarMatrix()` 函数求得的均值仍有用，因此此处没有做改动，此处改为调用求取全部行向量均值的函数也可行，同时也可以进一步提升性能。并将平均人脸向量 `mean` 写入训练结果文件 `model.json` 中。然后将样本中心化，这里使用的方法为，把 `mean` 纵向拼接成一个和 `f_samples` 一样大的矩阵 `meanMat` ，每一行都是原先的 `mean` 矩阵，这样直接将 `f_samples` 与 `meanMat` 做矩阵减法，即完成了中心化。

```c++
    //calculate the mean vector and then centralize
    Mat mean, cov;
    calcCovarMatrix(samples, cov, mean, COVAR_NORMAL | COVAR_ROWS, CV_64FC1);
    model<<"mean"<<mean;
    cout<<"the mean image is generated"<<endl;
    Mat meanMat = mean.clone();
    for(int i = 1; i < 205; i++) {
        vconcat(meanMat, mean, meanMat);
    }
    Mat f_samples;
    samples.convertTo(f_samples, CV_64FC1);
    Mat centralized;
    centralized = f_samples - meanMat;
    // cout<<"centralized size: "<<centralized.rows<<", "<<centralized.cols<<endl;
```

接下来这部分是输出生成的平均脸，不再赘述

```c++
    //reshape the mean vector to make it a image again
    Mat meanimg;
    mean.convertTo(meanimg, CV_8UC1);
    meanimg = meanimg.reshape(0, rows);
    imshow("mean", meanimg);
    waitKey(0);
```

接下来则是通过 SVD 计算特征脸，取 SVD 得到的右奇异矩阵的前 `p` 行作为特征脸矩阵 `pc` ，此处变量名为 `a` ，这里取前 `p` 行是由于 `SVD::compute()` 得到的奇异值矩阵中的奇异值是从上到下由大到小排列的，将其写入训练结果文件 `model.json` 中，并计算 `pc` 的转置 `pct`

```c++
    //calculate the eigenvalues and eigenvectors
    Mat eigenvals, eigenvecs;   //eigenvecs: 10304 rows and 10304 cols
    Mat left;
    SVD::compute(centralized, eigenvals, left, eigenvecs, SVD::FULL_UV);
    // cout<<"eigenval size: "<<eigenvals.rows<<", "<<eigenvals.cols<<endl;
    // cout<<"left size: "<<left.rows<<", "<<left.cols<<endl;
    // cout<<"eigenvec size: "<<eigenvecs.rows<<", "<<eigenvecs.cols<<endl;
    cout<<"eigenfaces generated"<<endl;
    
    int p;
    cout<<"please input the number of PCs you wanted"<<endl;
    cin>>p;
    Mat a = eigenvecs.rowRange(0, p).clone();
    model<<"pc"<<a;
    Mat at;
    transpose(a, at);
```

接下来为将前 10 张特征脸拼接显示出来。由于特征脸矩阵中取值的问题，直接转换为 `CV_8UC1` 格式输出的话会是一片黑色，因此这里先使用 `MINMAX` 算法将其范围调整至 `[0, 255]` ，再进行显示

```c++
    Mat show;
    for(int i = 0; i < 10; i++) {
        Mat temp = eigenvecs.rowRange(i, i+1).clone();
        normalize(temp, temp, 0, 255, NORM_MINMAX);
        temp.convertTo(temp, CV_8UC1);
        temp = temp.reshape(0, rows);
        resize(temp, temp, Size(184, 224));
        if(i == 0) {
            show = temp;
        }
        else {
            hconcat(show, temp, show);
        }
    }
    imshow("first 10 eigenfaces", show);
    waitKey(0);
```

最后则是将训练集映射到特征脸空间中，使用中心化后的训练集乘以特征脸矩阵的转置即可得到结果矩阵 `train_project` ，其中每一个行向量则对应一张图像降维后的结果，将映射后得到的结果也写入训练结果文件 `model.json` 中。至此则完成了模型的训练。

```c++
    cout<<"generating training set projection"<<endl;
    Mat train_project = centralized * at;   // each info is a row vec
    model<<"train_project"<<train_project;
    cout<<"done"<<endl;
```

##### 识别测试部分

此部分中，先将测试集转换为灰度图后进行直方图均衡化，然后从 `model.json` 中读取训练好的模型，包括平均人脸 `mean` ，训练集映射到特征脸空间的结果 `train_project` ，和特征人脸矩阵 `pc` 。其中 `mean` 是一个行向量，数据数与图片的像素数相同，`train_project` 是一个矩阵，有与 `training_set` 中图像数量相同的行数，和与选取的 `pc` 数量相同的列数，每一行代表一个 `training_set` 中图像投影到特征脸空间后的降维结果。然后调用 `transpose` 函数求出 `pc` 的转置 `pct` 。这样所需的训练结果数据就已经准备好了

```c++
void mytest(string model, vector<Mat> test_set)
{
    fstream ftxt("testing_result.txt", ios::out);
    ftxt.clear();
    //histogram equalization
    for(int i = 0; i < test_set.size(); i++) {
        cvtColor(test_set[i], test_set[i], COLOR_RGB2GRAY);
        equalizeHist(test_set[i], test_set[i]);
    }
    FileStorage fs(model, FileStorage::READ);
    Mat mean;
    fs["mean"]>>mean;
    Mat train_project;
    fs["train_project"]>>train_project;
    Mat pc, pct;
    fs["pc"]>>pc;
    transpose(pc, pct);
```

然后将测试集的图像每个转化为一个行向量，全部存入矩阵 `test_sample` 中，然后将每一行都减去 `mean` ，这里的实现方法是把 `mean` 纵向拼接成一个和 `test_sample` 一样大的矩阵 `test_mean` ，每一行都是原先的 `mean` 矩阵，这样直接将 `test_sample` 与 `test_mean` 做矩阵减法，即完成了中心化，存入 `test_centr` 中。然后则使用 `test_centr` 乘 `pct`（矩阵乘法），即可得到 `test_set` 中人脸投影到特征脸空间后的结果 `test_project` 。同样地，`test_project` 中每一行即为 `test_set` 中一个人脸投影到特征脸空间后的结果。

```c++
    Mat test_sample = test_set[0].reshape(0, 1);
    Mat test_mean = mean.clone();
    for(int i = 1; i < test_set.size(); i++) {
        vconcat(test_sample, test_set[i].reshape(0, 1), test_sample);
        vconcat(test_mean, mean, test_mean);
    }
    test_sample.convertTo(test_sample, CV_64FC1);
    // cout<<"test sample size: "<<test_sample.rows<<", "<<test_sample.cols<<endl;
    Mat test_centr = test_sample - test_mean;
    cout<<"generating testing set projection"<<endl;
    Mat test_project = test_centr * pct;
    cout<<"done"<<endl;
```

接下来则与训练集结果进行比较。对于 `test_project` 的每一行，计算其与 `train_project` 每行的欧氏距离，并记录 `train_project` 中欧式距离最小的行编号，将结果写入 `testing_result.txt` 中。通过 `i/5 + 1` 可以得到 `test_face` 对应的脸的编号，`i%5 + 6` 可以得到这是对应脸的第几张图片，而 `min_ind/5 + 1` 可以得到最相似的 `training_face` 中对应的脸的编号，`min_ind%5 + 1` 可以得到这是对应脸的第几张图片，将这些信息输出到 `testing_result.txt` 中。判断一次测试结果是否正确的方法即为判断 test face 与 result face 是否为同一张脸，若是，则表明结果正确，将 `correct_cnt` 自增 `1` ，反之则结果错误，不进行操作，将每次测试的正确与否也输入至 `testing_result.txt` 中。最后在循环结束后，`correct_cnt` 中的值即为测试结果正确数量，用 `correct_cnt/test_set.size()` 即可得到本次测试的正确率。将正确率也写入 `testing_result.txt` 中。

```c++
    // comparison
    int correct_cnt = 0;
    for(int i = 0; i < test_set.size(); i++) {
        // compute distance
        int min_ind;
        double min_dist;
        Mat vec_test = test_project.rowRange(i, i+1).clone();
        for(int index = 0; index < train_project.rows; index++) {
            Mat vec_train = train_project.rowRange(index, index+1).clone();
            Mat distance = vec_test - vec_train;
            pow(distance, 2, distance);
            double sum = 0;
            for(int j = 0; j < distance.cols; j++) {
                sum += distance.at<float64_t>(0, j);
            }
            if(index == 0) {
                min_dist = sum;
                min_ind = 0;
            }
            else {
                if(sum < min_dist) {
                    min_dist = sum;
                    min_ind = index;
                }
            }
        }
        ftxt<<"test face: s"<<i/5 + 1<<"p"<<i%5 + 6<<endl;
        ftxt<<"result face: s"<<min_ind/5 + 1<<"p"<<min_ind%5 + 1<<endl;
        ftxt<<"min distance = "<<min_dist<<endl;
        if(i/5 == min_ind/5) {
            correct_cnt++;
            ftxt<<"passed"<<endl;
        }
        else {
            ftxt<<"failed"<<endl;
        }
        ftxt<<"-------------------------"<<endl;
    }
    ftxt<<"acc = "<<(double)correct_cnt/test_set.size()<<endl;
    ftxt.close();
}
```

##### 图像重构部分

图像重构时同样也需要之前训练好的模型，因此从 `model.json` 中读取特征脸矩阵 `pc` 并计算其转置 `pct` 。将同样直方图均衡化并中心化后的图像（一个行向量）先乘以 `pct` ，得到其降维后的结果，然后乘以 `pc` 来重构图像，最后将这些重构出的图像放大两倍拼接起来显示出来。

```c++
void myreconstruction(string model, vector<Mat> reconstruct_set)
{
    FileStorage fs(model, FileStorage::READ);
    Mat pc;
    fs["pc"]>>pc;
    Mat pct;
    transpose(pc, pct);
    Mat mean;
    fs["mean"]>>mean;
    Mat result;
    Mat src;
    for(int i = 0; i < 10; i++) {
        cvtColor(reconstruct_set[i], reconstruct_set[i], COLOR_RGB2GRAY);
        equalizeHist(reconstruct_set[i], reconstruct_set[i]);
        Mat temp = reconstruct_set[i].reshape(0, 1);
        temp.convertTo(temp, CV_64FC1);
        temp = temp - mean;
        temp = temp * pct;
        temp = temp * pc;
        temp = temp + mean;
        temp = temp.reshape(0, rows);
        resize(temp, temp, Size(184, 224));
        if(i == 0) {
            result = temp;
            src = reconstruct_set[i];
        }
        else {
            hconcat(result, temp, result);
            hconcat(src, reconstruct_set[i], src);
        }
    }
    normalize(result, result, 0, 255, NORM_MINMAX);
    normalize(src, src, 0, 255, NORM_MINMAX);
    result.convertTo(result, CV_8UC1);
    src.convertTo(src, CV_8UC1);
    imshow("reconstruct", result);
    imshow("src", src);
    waitKey(0);
}
```

#### 
