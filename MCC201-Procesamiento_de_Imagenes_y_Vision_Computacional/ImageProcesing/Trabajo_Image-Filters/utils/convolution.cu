# include <cuda.h>
# include <cuda_runtime.h>
extern "C"
unsigned char * convolutionGPU(unsigned char * rowdata, int width, int height, signed char * kernell, int kernelSize, int nchann);

__global__ void convolute_1PixelGPU(unsigned char * data_dev, unsigned char * convData_dev, int width, int height, signed char * kernellDev, int kernelSize, int nchann){
    int posThread = blockIdx.x*blockDim.x + threadIdx.x;
    if(posThread < width*height*nchann){
        //Compute the sum of the kernell elements
        int sumKernell = 0;
        for(int i = 0; i < kernelSize*kernelSize; i++)
            sumKernell += kernellDev[i];
        sumKernell = (sumKernell <= 0) ? 1: sumKernell;

        // Start to convolute:
        int acumm = 0;
        int midSZ = (kernelSize - 1)/2;
        for(int ky = 0; ky < kernelSize; ky++){
            for(int kx = 0; kx < kernelSize; kx++){
                int pdata = posThread + nchann*width*(ky - midSZ) - midSZ*nchann + kx*nchann;
                if(pdata > 0 && pdata < width*height*nchann)
                    acumm += data_dev[pdata]*kernellDev[kx + ky*kernelSize];
            }
        }
        int ans = acumm/sumKernell;
        if(ans > 255)
            ans = 255;
        if(ans < 0)
            ans = 0;
        convData_dev[posThread] =(unsigned char) ans;
    }
}

unsigned char * convolutionGPU(unsigned char * rowdata, int width, int height, signed char * kernell, int kernelSize, int nchann){
    int size = nchann*width*height;
    unsigned char * convolutedData = new unsigned char[size];
    unsigned char * dataDev;
    unsigned char * convDataDev;
    signed char * kernellDev;

    cudaMalloc((void**)&dataDev, size*sizeof(unsigned char));
    cudaMalloc((void**)&convDataDev, size*sizeof(unsigned char));
    cudaMalloc((void**)&kernellDev, kernelSize*kernelSize*sizeof(signed char));
    cudaMemcpy(dataDev, rowdata, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(kernellDev, kernell, kernelSize*kernelSize*sizeof(signed char), cudaMemcpyHostToDevice);


    int nThreads = 1024;
    int nBlocks = (size % nThreads > 0)? size/nThreads + 1: size/nThreads;
    convolute_1PixelGPU<<<nBlocks, nThreads>>>(dataDev, convDataDev, width, height, kernellDev, kernelSize, nchann);
    cudaMemcpy(convolutedData, convDataDev, size*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(dataDev);
    cudaFree(convDataDev);
    cudaFree(kernellDev);
    return convolutedData;
}
