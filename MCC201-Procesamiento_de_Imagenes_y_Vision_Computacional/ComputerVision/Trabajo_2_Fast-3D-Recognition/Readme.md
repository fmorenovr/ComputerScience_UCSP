## Fast 3D Recognition

We try to reproduce the paper adjunt in this repository. To do this, we use PCL library and Kinect to get data and parameters.

## Algorithm

We need to get Viewpoint Feature Histogram which define a viewpoint direction component and
a surface shape component comprised of an extended FPFH (based on description of the object).

  The algorithm to calculate FPFH(Fast Point Feature Histograms):
  
  ``` go
     for each point p in cloud P

      1. pass 1:

         1. get the nearest neighbors of p

         2. for each pair of `p, p_k` (where `p_k` is a neighbor of `p`, compute the three angular values

         3. bin all the results in an output SPFH histogram

      2. pass 2:

         1. get the nearest neighbors of `p`

         3. use each SPFH of `p` with a weighting scheme to assemble the FPFH of `p`:
  ```

  SPFH (Simplified Point Feature Histogram) is the simple process to calculate orientation between point p and p_k neighbors.

## Run the project

You will need to connect and verify status of your kinect camera (or your multiples kinects) and then follow the next steps:

* First, Go to the `fast-3D-recognition` directory and check the Readm file to install all dependencies.

* Once installed all dependencies, just run the following command:

      ./compileFile

