# include <vector>
# include "convolucion.h"
# include <math.h>

unsigned char * RGB2Gray(unsigned char * dataRGB, int size){
    int szGray = size/3;
    // Since human is more sensible to Green value it have more predominance 
    double Rp = 0.2126;
    double Gp = 0.7152;
    double Bp = 0.0722;
    unsigned char * dataGray = new unsigned char[szGray];
    for(int i = 0, k = 0; i < size; i+=3, k++){
        double val= Rp*(double)dataRGB[i] + Gp*(double)dataRGB[i+1] + Bp*(double)dataRGB[i+2];
        dataGray[k] = (unsigned char)val;
    }
    return dataGray;
}

unsigned char EvaluatePolynomial(const std::vector<double> & pol, unsigned char pixel);
unsigned char * PolinomialTransform(const std::vector<double> & coeff, unsigned char * data, int width, int height, int nchann){
    int size = nchann*height*width;
    if( size > 0){
        unsigned char * dataFiltered = new unsigned char[size];
        for(int i = 0; i < size; i++){
            dataFiltered[i] = EvaluatePolynomial(coeff, data[i]);
        }
        return dataFiltered;
    }
    else
        return nullptr;
}

double PowChar(unsigned char v, int potence){
    if(potence == 0 ){
        return 1.0;
    }
    return (double)v*PowChar(v, potence -1);
}
unsigned char EvaluatePolynomial(const std::vector<double> & pol, unsigned char pixel){
    double val = 0;
    for(int i = 0, pot = pol.size() - 1; i < pol.size(); i++, pot--){
        val += pol[i]*PowChar(pixel, pot);
    }
    int ans = (int) val;
    if( ans > 255)
        ans = 255;
    if (ans < 0)
        ans = 0;
    return (unsigned char)ans;
}

unsigned char * DoubleThreshold(unsigned char * dataGray, int size, int lThres, int hThres){
    unsigned char * thresholed = new unsigned char[size];
    for(int i = 0; i < size; i++){
        if(dataGray[i] < lThres)
            thresholed[i] = 0;
        else if (dataGray[i] > hThres)
            thresholed[i] = 255;
        else
            thresholed[i] = dataGray[i];

    }
    return thresholed;
}
unsigned char * ThresHoldFilter(unsigned char * dataGray, int size, unsigned char thres){
    unsigned char * thresholed = new unsigned char[size];
    for(int i = 0; i < size; i++){
        if(dataGray[i] > thres)
            thresholed[i] = 255;
        else
            thresholed[i] = dataGray[i];
    }
    return thresholed;
}
unsigned char * InvertGrayScale(unsigned char * dataGray, int size){
    unsigned char * inverted = new unsigned char[size];
    for(int i = 0; i < size; i++){
        inverted[i] = 255 - dataGray[i];
    }
    return inverted;
}
unsigned char * arctanOperator(unsigned char * gx, unsigned char * gy, int size){
    unsigned char * result = new unsigned char[size];
    for(int i = 0; i < size; i++){
        result[i] =(unsigned char) (atan2((double)gy[i],(double)gx[i])*127) +128;
    }
    return result;
}
unsigned char * LaplaceFilter(unsigned char * graydata, int w, int h){
    signed char LaplacianKernell[25] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 24, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    unsigned char * laplace = convolution(graydata, w, h, LaplacianKernell, 5, 1);
    return laplace;
}
// Return Edge detector in GrayScale 8bits x pixel
// As an input get the image in RGB format
// The method uses a Gaussian Filter, followed by a Laplacian Filter & threshold filter.
unsigned char * SimpleEdgeDetector(unsigned char * rowdata, int width, int height, int lthres, int hthres){
    signed char GaussianKernell[25] = {2, 4, 5, 4, 2, 4, 9, 12, 9, 4, 5, 12, 15, 12, 5, 4, 9, 12, 9, 4, 2, 4, 5, 4, 2};

    //signed char GxKernell[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    //signed char GyKernell[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    unsigned char * dataGray = RGB2Gray(rowdata, width*height*3);
    unsigned char * dataGauss = convolution(dataGray, width, height, GaussianKernell, 5, 1);
    /*unsigned char * Gx = convolution(dataGauss, width, height, GxKernell, 3, 1);
    unsigned char * Gy = convolution(dataGauss, width, height, GyKernell, 3, 1);
    unsigned char * atanData = arctanOperator(Gx, Gy, width*height);
    delete [] dataGray;
    delete [] dataGauss;
    delete [] Gx;
    delete [] Gy;
    return atanData;*/
    unsigned char * laplacian = LaplaceFilter(dataGauss, width, height);
    unsigned char * thresholded = ThresHoldFilter(laplacian, height*width, 150);
    unsigned char * invertedEdges = InvertGrayScale(thresholded, width*height);
    delete [] dataGray;
    delete [] dataGauss;
    delete [] laplacian;
    delete [] thresholded;
    return invertedEdges;

}
