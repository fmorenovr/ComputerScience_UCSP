# include <iostream>
// Works for pixels entrelazed RGB
unsigned char convolute1Pixel(unsigned char * rowdata, int width, int height, signed char * kernell, int kernelSize, int pos, int nchann){
    int sumKernell = 0;
    for(int i = 0; i < kernelSize*kernelSize; i++)
        sumKernell+=kernell[i];
    if(sumKernell <= 0)
        sumKernell = 1;

    int acumm = 0;
    int midSZ = (kernelSize - 1)/2;
    for(int ky = 0; ky < kernelSize; ky++){
        for(int kx = 0; kx < kernelSize; kx++){
            int pdata = pos + nchann*width*(ky - midSZ) - midSZ*nchann + kx*nchann;
            if(pdata >= 0 && pdata < width*height*nchann)
                acumm += rowdata[pdata]*kernell[kx + ky*kernelSize];
        }
    }
    int ans = acumm/sumKernell;
    if(ans > 255)
        ans = 255;
    if(ans < 0)
        ans = 0;
    return (unsigned char) ans;

}
unsigned char * convolution(unsigned char * rowdata, int width, int height, signed char * kernell, int kernelSize, int nchann){
    int size = nchann*width*height;
    unsigned char * gendata = new unsigned char[size];
    for(int i = 0; i < size; i++){
        gendata[i] = convolute1Pixel(rowdata, width, height, kernell, kernelSize, i, nchann);
    }
    return gendata;
}


